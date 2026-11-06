"""Deterministic after-sales rules and idempotent write execution.

The LLM can only *prepare* a return.  Every eligibility check and the actual
database transaction live here, on the server.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.exc import IntegrityError

from app.config import Config
from app.extensions import db
from app.models.agent import PendingAction
from app.models.commerce import Order, OrderItem, ReturnRequest
from app.services.return_policy import ReturnDecision, ReturnPolicyEngine


class AfterSalesError(Exception):
    """A business-rule failure with a user-safe message."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


class AfterSalesService:
    """Prepare and commit return actions."""

    @staticmethod
    def prepare_return(
        *,
        order: Order,
        item_ids: list[int],
        reason: str,
        reason_code: str,
        condition: str,
        user_id: int,
        session_id: int,
        run_id: str,
        tool_call_id: str | None = None,
    ) -> PendingAction:
        item_ids = sorted({int(item_id) for item_id in item_ids})
        if not item_ids:
            raise AfterSalesError("INVALID_ARGUMENTS", "请选择要退货的商品")
        if order.user_id != user_id:
            raise AfterSalesError("NOT_FOUND", "未找到该订单，请确认订单号")

        now = datetime.now()
        items = (
            OrderItem.query.filter(OrderItem.order_id == order.id, OrderItem.id.in_(item_ids))
            .order_by(OrderItem.id)
            .all()
        )
        found = {item.id for item in items}
        if found != set(item_ids):
            raise AfterSalesError("INVALID_ARGUMENTS", "部分商品不属于该订单，请重新选择")

        try:
            decision = ReturnPolicyEngine.evaluate(
                order=order,
                items=items,
                condition=condition,
                reason_code=reason_code,
                now=now,
            )
        except ValueError as exc:
            raise AfterSalesError("INVALID_ARGUMENTS", str(exc)) from exc
        if not decision.eligible:
            raise AfterSalesError(decision.code, decision.message)

        refund_amount = round(sum(item.unit_price * item.quantity for item in items), 2)
        reason_text = (reason or "").strip()[:500]
        item_details = [
            {
                "item_id": item.id,
                "sku_id": item.sku_id,
                "product_name": item.product_name,
                "quantity": item.quantity,
                "unit_price": round(item.unit_price, 2),
                "refund": round(item.unit_price * item.quantity, 2),
            }
            for item in items
        ]

        intent_payload = {
            "order_id": order.id,
            "order_no": order.order_no,
            "item_ids": item_ids,
            "items": item_details,
            "reason": reason_text,
            "reason_code": reason_code,
            "condition": condition,
            "refund_amount": refund_amount,
        }
        args_hash = _sha256(intent_payload)
        idempotency_key = _sha256(
            {
                "user_id": user_id,
                "action_type": "return",
                "order_id": order.id,
                "item_ids": item_ids,
                "args_hash": args_hash,
            }
        )
        payload = {
            **intent_payload,
            "policy_decision": decision.to_dict(),
            "previewed_at": now.isoformat(timespec="seconds"),
        }

        existing = PendingAction.query.filter_by(idempotency_key=idempotency_key).first()
        if existing:
            if existing.status == "pending":
                # Re-submitting the same intent while a card is open must not
                # create a second pending action.
                return existing
            if existing.status in {"cancelled", "expired", "failed"}:
                # The user cancelled, the card expired, or a previous attempt
                # failed.  This is a new attempt, not a duplicate write.
                idempotency_key = _sha256(
                    {"base": idempotency_key, "retry": uuid.uuid4().hex}
                )
            else:
                # executing / success: never create a second write intent.
                raise AfterSalesError(
                    "DUPLICATE_REQUEST",
                    "该退货申请正在处理或已完成，请勿重复提交",
                )

        expires_at = now + timedelta(minutes=Config.AGENT_ACTION_TTL_MINUTES)
        policy_versions = ",".join(decision.policy_versions) or None
        display = {
            "action_type": "return",
            "title": "确认退货",
            "summary": (
                f"订单 {order.order_no} 提交退货申请 {len(items)} 件，"
                f"预计退款 ¥{refund_amount:.2f}"
            ),
            "details": item_details,
            "condition": condition,
            "reason_code": reason_code,
            "policy_version": policy_versions,
            "decision_code": decision.code,
            "requires_review": decision.requires_review,
            "policy_source": sorted(
                {
                    item.policy_snapshot.get("source_document", "")
                    for item in decision.items
                    if item.policy_snapshot
                }
            ),
            "notice": (
                "确认后仅提交退货申请，审核、退款和物流处置以后续状态为准；"
                "取消不会产生任何变更。"
            ),
            "expires_at": expires_at.isoformat(timespec="seconds"),
        }

        pending = PendingAction(
            run_id=run_id,
            user_id=user_id,
            session_id=session_id,
            action_type="return",
            resource_type="order",
            resource_id=order.id,
            payload_json=payload,
            display_json=display,
            args_hash=args_hash,
            status="pending",
            idempotency_key=idempotency_key,
            tool_call_id=tool_call_id,
            expires_at=expires_at,
        )
        db.session.add(pending)
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            existing = PendingAction.query.filter_by(idempotency_key=idempotency_key).first()
            if existing and existing.status == "pending":
                return existing
            raise AfterSalesError("DUPLICATE_REQUEST", "相同退货申请已处理或已失效，请重新发起")
        return pending

    @staticmethod
    def commit_return(pending: PendingAction) -> dict[str, Any]:
        """Execute the confirmed return exactly once."""
        existing = ReturnRequest.query.filter_by(idempotency_key=pending.idempotency_key).first()
        if existing:
            return AfterSalesService._serialize_return(existing)

        payload = pending.payload_json or {}
        order = db.session.get(Order, payload.get("order_id"))
        if not order or order.user_id != pending.user_id:
            raise AfterSalesError("NOT_FOUND", "订单不存在或无权操作")

        item_ids = payload.get("item_ids") or []
        items = (
            OrderItem.query.filter(OrderItem.order_id == order.id, OrderItem.id.in_(item_ids))
            .order_by(OrderItem.id)
            .all()
        )
        if {item.id for item in items} != set(item_ids):
            raise AfterSalesError("INVALID_ARGUMENTS", "退货商品与订单不一致")

        expected_decision = (payload.get("policy_decision") or {})
        try:
            current_decision = ReturnPolicyEngine.evaluate(
                order=order,
                items=items,
                condition=str(payload.get("condition") or ""),
                reason_code=str(payload.get("reason_code") or ""),
            )
        except ValueError as exc:
            raise AfterSalesError("CONFIRMATION_STALE", "确认内容已失效，请重新发起") from exc
        if not current_decision.eligible:
            raise AfterSalesError(current_decision.code, current_decision.message)
        if not _decision_matches_preview(expected_decision, current_decision):
            raise AfterSalesError(
                "CONFIRMATION_STALE",
                "订单、商品状态或退货政策已发生变化，请重新发起并确认",
            )

        current_refund = round(sum(item.unit_price * item.quantity for item in items), 2)
        if abs(current_refund - float(payload.get("refund_amount") or 0)) > 0.001:
            raise AfterSalesError(
                "CONFIRMATION_STALE",
                "订单金额已发生变化，请重新发起并确认",
            )

        policy_versions = ",".join(current_decision.policy_versions) or None
        return_request = ReturnRequest(
            order_id=order.id,
            user_id=pending.user_id,
            item_ids_json=item_ids,
            reason=payload.get("reason") or "",
            refund_amount=current_refund,
            status=(
                "manual_review"
                if current_decision.requires_review
                else "submitted"
            ),
            policy_version=policy_versions,
            decision_json=current_decision.to_dict(),
            idempotency_key=pending.idempotency_key,
        )
        db.session.add(return_request)
        for item in items:
            item.item_status = "returning"

        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            existing = ReturnRequest.query.filter_by(idempotency_key=pending.idempotency_key).first()
            if existing:
                return AfterSalesService._serialize_return(existing)
            raise

        return AfterSalesService._serialize_return(return_request)

    @staticmethod
    def _serialize_return(return_request: ReturnRequest) -> dict[str, Any]:
        status_text = (
            "已提交并进入人工核验"
            if return_request.status == "manual_review"
            else "已提交"
        )
        return {
            "return_id": return_request.id,
            "return_no": f"R{return_request.id:08d}",
            "order_id": return_request.order_id,
            "refund_amount": round(return_request.refund_amount, 2),
            "status": return_request.status,
            "policy_version": return_request.policy_version,
            "display_text": (
                f"退货申请{status_text}（单号 R{return_request.id:08d}），"
                f"预计退款 ¥{return_request.refund_amount:.2f}；"
                "审核与退款进度请以后续通知为准。"
            ),
        }

    @staticmethod
    def confirmation_token(pending: PendingAction) -> str:
        """Return an HMAC bound to the user, action payload and expiry.

        The raw token is never stored. It is deterministic for one pending
        action so a restored confirmation card can receive the same token.
        """
        expires = (
            pending.expires_at.isoformat(timespec="seconds")
            if pending.expires_at
            else ""
        )
        message = "|".join(
            [
                "v1",
                str(pending.id),
                str(pending.user_id),
                str(pending.session_id),
                pending.args_hash,
                expires,
            ]
        )
        return hmac.new(
            Config.AGENT_ACTION_SECRET.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def verify_confirmation_token(
        pending: PendingAction,
        token: str | None,
    ) -> bool:
        if not token:
            return False
        expected = AfterSalesService.confirmation_token(pending)
        return hmac.compare_digest(expected, str(token))


def _decision_matches_preview(
    expected: dict[str, Any],
    current: ReturnDecision,
) -> bool:
    """Compare the business meaning of a preview and a fresh re-evaluation."""
    expected_items = expected.get("items") or []
    current_items = [item.to_dict() for item in current.items]
    if len(expected_items) != len(current_items):
        return False
    for expected_item, current_item in zip(expected_items, current_items):
        for field in (
            "item_id",
            "sku_id",
            "category",
            "eligible",
            "requires_review",
            "code",
            "policy_version",
            "policy_snapshot",
        ):
            if expected_item.get(field) != current_item.get(field):
                return False
    return (
        expected.get("condition") == current.condition
        and expected.get("reason_code") == current.reason_code
        and expected.get("policy_versions") == current.policy_versions
    )
