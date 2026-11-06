"""Versioned return policy selection and deterministic eligibility decisions.

The LLM may explain a retrieved policy, but it must never decide whether a hard
business rule passes. This module owns that decision and returns an auditable
snapshot that is persisted with the pending action and final return request.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.extensions import db
from app.models.commerce import Order, OrderItem, ReturnPolicy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILTIN_POLICY_FILE = PROJECT_ROOT / "data" / "return_policies.json"

ITEM_CONDITIONS = {"unopened", "opened", "activated", "used", "damaged"}
REASON_CODES = {
    "no_longer_needed",
    "size_or_fit",
    "damaged_on_arrival",
    "defective_quality",
    "wrong_item",
    "other",
}
QUALITY_REASON_CODES = {
    "damaged_on_arrival",
    "defective_quality",
    "wrong_item",
}


@dataclass
class ItemPolicyDecision:
    item_id: int
    sku_id: str
    product_name: str
    category: str
    eligible: bool
    requires_review: bool
    code: str
    message: str
    policy_version: str | None = None
    policy_rule_id: int | None = None
    policy_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReturnDecision:
    eligible: bool
    requires_review: bool
    code: str
    message: str
    policy_versions: list[str]
    evaluated_at: str
    condition: str
    reason_code: str
    items: list[ItemPolicyDecision]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReturnPolicyEngine:
    """Evaluate all return eligibility rules in application code."""

    @staticmethod
    def seed_builtin_policies(*, path: Path = BUILTIN_POLICY_FILE) -> int:
        """Upsert the demo policy file into the executable policy table."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = str(payload["policy_version"])
        effective_from = datetime.fromisoformat(payload["effective_from"])
        source_document = str(payload["source_document"])

        changed = 0
        categories: list[str] = []
        for rule in payload.get("rules", []):
            category = str(rule["category"])
            categories.append(category)
            policy = ReturnPolicy.query.filter_by(
                policy_version=version,
                category=category,
            ).first()
            if not policy:
                policy = ReturnPolicy(
                    policy_version=version,
                    category=category,
                    category_label=str(rule["category_label"]),
                    effective_from=effective_from,
                    source_document=source_document,
                    source_section=str(rule["source_section"]),
                )
                db.session.add(policy)
            policy.category_label = str(rule["category_label"])
            policy.effective_from = effective_from
            policy.return_window_days = int(rule["return_window_days"])
            policy.no_reason_allowed = bool(rule["no_reason_allowed"])
            policy.no_reason_requires_unopened = bool(
                rule["no_reason_requires_unopened"]
            )
            policy.manual_review_allowed = bool(rule["manual_review_allowed"])
            policy.source_document = source_document
            policy.source_section = str(rule["source_section"])
            policy.active = True
            changed += 1
        if categories:
            ReturnPolicy.query.filter(
                ReturnPolicy.policy_version == version,
                ReturnPolicy.category.notin_(categories),
            ).update({"active": False}, synchronize_session=False)
        db.session.commit()
        return changed

    @staticmethod
    def _select_policy(
        category: str,
        *,
        policy_as_of: datetime,
    ) -> ReturnPolicy | None:
        candidates = (
            ReturnPolicy.query.filter(
                ReturnPolicy.active.is_(True),
                ReturnPolicy.category.in_([category, "*"]),
                ReturnPolicy.effective_from <= policy_as_of,
            )
            .order_by(ReturnPolicy.effective_from.desc(), ReturnPolicy.id.desc())
            .all()
        )
        candidates = [
            policy
            for policy in candidates
            if policy.effective_to is None or policy.effective_to > policy_as_of
        ]
        exact = [policy for policy in candidates if policy.category == category]
        fallback = [policy for policy in candidates if policy.category == "*"]
        return (exact or fallback or [None])[0]

    @staticmethod
    def _snapshot(policy: ReturnPolicy) -> dict[str, Any]:
        return {
            "id": policy.id,
            "policy_version": policy.policy_version,
            "category": policy.category,
            "category_label": policy.category_label,
            "effective_from": policy.effective_from.isoformat(timespec="seconds"),
            "effective_to": (
                policy.effective_to.isoformat(timespec="seconds")
                if policy.effective_to
                else None
            ),
            "return_window_days": policy.return_window_days,
            "no_reason_allowed": policy.no_reason_allowed,
            "no_reason_requires_unopened": policy.no_reason_requires_unopened,
            "manual_review_allowed": policy.manual_review_allowed,
            "source_document": policy.source_document,
            "source_section": policy.source_section,
        }

    @classmethod
    def evaluate(
        cls,
        *,
        order: Order,
        items: list[OrderItem],
        condition: str,
        reason_code: str,
        now: datetime | None = None,
    ) -> ReturnDecision:
        if condition not in ITEM_CONDITIONS:
            raise ValueError(f"unsupported item condition: {condition}")
        if reason_code not in REASON_CODES:
            raise ValueError(f"unsupported reason code: {reason_code}")

        evaluated_at = now or datetime.now()
        decisions: list[ItemPolicyDecision] = []
        issues: list[str] = []
        policy_versions: set[str] = set()

        for item in items:
            base = {
                "item_id": item.id,
                "sku_id": item.sku_id,
                "product_name": item.product_name,
                "category": item.category,
            }
            if item.item_status != "normal":
                message = f"商品「{item.product_name}」已处于退货流程中"
                issues.append(message)
                decisions.append(
                    ItemPolicyDecision(
                        **base,
                        eligible=False,
                        requires_review=False,
                        code="ITEM_ALREADY_IN_RETURN",
                        message=message,
                    )
                )
                continue

            if order.status != "delivered" or not order.delivered_at:
                message = "订单尚未签收，暂不支持在线提交退货申请"
                issues.append(message)
                decisions.append(
                    ItemPolicyDecision(
                        **base,
                        eligible=False,
                        requires_review=False,
                        code="ORDER_NOT_DELIVERED",
                        message=message,
                    )
                )
                continue

            policy = cls._select_policy(item.category, policy_as_of=order.delivered_at)
            if not policy:
                message = (
                    f"商品「{item.product_name}」没有可适用的有效退货政策，"
                    "请转人工客服确认"
                )
                issues.append(message)
                decisions.append(
                    ItemPolicyDecision(
                        **base,
                        eligible=False,
                        requires_review=False,
                        code="POLICY_UNAVAILABLE",
                        message=message,
                    )
                )
                continue

            snapshot = cls._snapshot(policy)
            policy_versions.add(policy.policy_version)
            quality_issue = (
                reason_code in QUALITY_REASON_CODES or condition == "damaged"
            )

            if not quality_issue and not policy.no_reason_allowed:
                message = (
                    f"商品「{item.product_name}」属于{policy.category_label}，"
                    "不支持无理由退货；如属质量问题请转人工处理"
                )
                issues.append(message)
                decisions.append(
                    ItemPolicyDecision(
                        **base,
                        eligible=False,
                        requires_review=False,
                        code="NO_REASON_NOT_ALLOWED",
                        message=message,
                        policy_version=policy.policy_version,
                        policy_rule_id=policy.id,
                        policy_snapshot=snapshot,
                    )
                )
                continue

            if quality_issue and not policy.manual_review_allowed:
                message = (
                    f"商品「{item.product_name}」不支持在线退货申请，"
                    "请转人工客服处理"
                )
                issues.append(message)
                decisions.append(
                    ItemPolicyDecision(
                        **base,
                        eligible=False,
                        requires_review=False,
                        code="MANUAL_REVIEW_NOT_ALLOWED",
                        message=message,
                        policy_version=policy.policy_version,
                        policy_rule_id=policy.id,
                        policy_snapshot=snapshot,
                    )
                )
                continue

            deadline = order.delivered_at + timedelta(
                days=policy.return_window_days
            )
            if evaluated_at > deadline:
                message = (
                    f"商品「{item.product_name}」已超过 {policy.return_window_days} 天"
                    "退货申请期；如主张质量问题，请转人工客服核验"
                )
                issues.append(message)
                decisions.append(
                    ItemPolicyDecision(
                        **base,
                        eligible=False,
                        requires_review=False,
                        code="RETURN_WINDOW_EXPIRED",
                        message=message,
                        policy_version=policy.policy_version,
                        policy_rule_id=policy.id,
                        policy_snapshot=snapshot,
                    )
                )
                continue

            if (
                not quality_issue
                and policy.no_reason_requires_unopened
                and condition != "unopened"
            ):
                message = (
                    f"商品「{item.product_name}」已拆封或使用，"
                    "不满足无理由退货条件；如属质量问题请转人工客服核验"
                )
                issues.append(message)
                decisions.append(
                    ItemPolicyDecision(
                        **base,
                        eligible=False,
                        requires_review=False,
                        code="ITEM_CONDITION_NOT_ELIGIBLE",
                        message=message,
                        policy_version=policy.policy_version,
                        policy_rule_id=policy.id,
                        policy_snapshot=snapshot,
                    )
                )
                continue

            requires_review = bool(quality_issue)
            code = "MANUAL_REVIEW_REQUIRED" if requires_review else "ELIGIBLE"
            message = (
                "符合提交条件，提交后需人工核验商品问题"
                if requires_review
                else "符合在线提交退货申请的条件"
            )
            decisions.append(
                ItemPolicyDecision(
                    **base,
                    eligible=True,
                    requires_review=requires_review,
                    code=code,
                    message=message,
                    policy_version=policy.policy_version,
                    policy_rule_id=policy.id,
                    policy_snapshot=snapshot,
                )
            )

        eligible = bool(decisions) and all(item.eligible for item in decisions)
        requires_review = eligible and any(
            item.requires_review for item in decisions
        )
        if not decisions:
            code = "NO_ITEMS"
            message = "请选择要退货的商品"
        elif not eligible:
            code = next(item.code for item in decisions if not item.eligible)
            message = "；".join(dict.fromkeys(issues))
        elif requires_review:
            code = "MANUAL_REVIEW_REQUIRED"
            message = "符合提交条件，提交后将进入人工核验"
        else:
            code = "ELIGIBLE"
            message = "符合在线提交退货申请的条件"

        return ReturnDecision(
            eligible=eligible,
            requires_review=requires_review,
            code=code,
            message=message,
            policy_versions=sorted(policy_versions),
            evaluated_at=evaluated_at.isoformat(timespec="seconds"),
            condition=condition,
            reason_code=reason_code,
            items=decisions,
        )
