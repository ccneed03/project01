"""Deterministic after-sales rules and idempotent write execution.

The LLM can only *prepare* a return.  Every eligibility check and the actual
database transaction live here, on the server.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.exc import IntegrityError

from app.config import Config
from app.extensions import db
from app.models.agent import PendingAction
from app.models.commerce import Order, OrderItem, ReturnRequest

# Demo rule set. Move to a table when the policy needs to be edited at runtime.
NON_RETURNABLE_CATEGORIES = {"定制", "内衣", "虚拟商品"}
NON_RETURNABLE_CATEGORIES_LABEL = {
    "定制": "定制类商品",
    "内衣": "贴身衣物",
    "虚拟商品": "虚拟商品",
}


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
        if order.status != "delivered" or not order.delivered_at:
            raise AfterSalesError("NOT_ELIGIBLE", "订单尚未签收，暂不支持退货")
        if now - order.delivered_at > timedelta(days=Config.AGENT_RETURN_WINDOW_DAYS):
            raise AfterSalesError(
                "NOT_ELIGIBLE",
                f"订单已超过 {Config.AGENT_RETURN_WINDOW_DAYS} 天无理由退货期",
            )

        items = (
            OrderItem.query.filter(OrderItem.order_id == order.id, OrderItem.id.in_(item_ids))
            .order_by(OrderItem.id)
            .all()
        )
        found = {item.id for item in items}
        if found != set(item_ids):
            raise AfterSalesError("INVALID_ARGUMENTS", "部分商品不属于该订单，请重新选择")

        for item in items:
            if item.item_status != "normal":
                raise AfterSalesError("NOT_ELIGIBLE", f"商品「{item.product_name}」已处于退货流程中")
            if item.category in NON_RETURNABLE_CATEGORIES:
                raise AfterSalesError(
                    "NOT_ELIGIBLE",
                    f"商品「{item.product_name}」属于{NON_RETURNABLE_CATEGORIES_LABEL.get(item.category, item.category)}，不支持无理由退货",
                )

        refund_amount = round(sum(item.unit_price * item.quantity for item in items), 2)
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

        payload = {
            "order_id": order.id,
            "order_no": order.order_no,
            "item_ids": item_ids,
            "items": item_details,
            "reason": (reason or "").strip()[:500],
            "refund_amount": refund_amount,
        }
        args_hash = _sha256(payload)
        idempotency_key = _sha256(
            {
                "user_id": user_id,
                "action_type": "return",
                "order_id": order.id,
                "item_ids": item_ids,
                "args_hash": args_hash,
            }
        )

        existing = PendingAction.query.filter_by(idempotency_key=idempotency_key).first()
        if existing:
            if existing.status == "pending":
                return existing
            raise AfterSalesError("DUPLICATE_REQUEST", "相同退货申请已处理或已失效，请重新发起")

        expires_at = now + timedelta(minutes=Config.AGENT_ACTION_TTL_MINUTES)
        display = {
            "action_type": "return",
            "title": "确认退货",
            "summary": f"订单 {order.order_no} 退货 {len(items)} 件，预计退款 ¥{refund_amount:.2f}",
            "details": item_details,
            "notice": "确认后才会提交退货申请；取消不会产生任何变更。",
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
        for item in items:
            if item.item_status != "normal":
                raise AfterSalesError("ALREADY_EXECUTED", "该商品已在退货流程中")

        return_request = ReturnRequest(
            order_id=order.id,
            user_id=pending.user_id,
            item_ids_json=item_ids,
            reason=payload.get("reason") or "",
            refund_amount=float(payload.get("refund_amount") or 0),
            status="approved",
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
        return {
            "return_id": return_request.id,
            "return_no": f"R{return_request.id:08d}",
            "order_id": return_request.order_id,
            "refund_amount": round(return_request.refund_amount, 2),
            "status": return_request.status,
            "display_text": (
                f"退货申请已提交（单号 R{return_request.id:08d}），"
                f"退款 ¥{return_request.refund_amount:.2f} 将在 1-3 个工作日原路退回。"
            ),
        }
