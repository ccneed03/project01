"""Shared helpers for order-related tools."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from app.agents.context import AgentContext
from app.agents.schemas import AgentState
from app.extensions import db
from app.models.commerce import Order

ORDER_STATUS_LABELS = {
    "paid": "已付款",
    "shipped": "已发货",
    "delivered": "已签收",
    "cancelled": "已取消",
}

ITEM_STATUS_LABELS = {
    "normal": "正常",
    "returning": "退货处理中",
    "returned": "已退货",
}


def iso(value: datetime | None) -> str | None:
    return value.isoformat(timespec="seconds") if value else None


def mask_phone(phone: str | None) -> str:
    """Mask a phone number before it reaches the model or the UI."""
    if not phone:
        return ""
    if len(phone) < 7:
        return "***"
    return f"{phone[:3]}****{phone[-4:]}"


def resolve_order(order_no: str | None, context: AgentContext, state: AgentState) -> Order | None:
    """Resolve an order using server-side ownership checks.

    The model may only provide an order number.  ``user_id`` always comes from
    the trusted context, so passing someone else's order number simply looks
    like a non-existent order.
    """
    if order_no:
        order = Order.query.filter_by(order_no=order_no, user_id=context.user_id).first()
        if order:
            state.active_order_id = order.id
            state.candidate_orders = [order.id]
        return order

    if state.active_order_id:
        order = db.session.get(Order, state.active_order_id)
        if order and order.user_id == context.user_id:
            return order
    return None


def order_data(order: Order) -> dict[str, Any]:
    """Build the safe, non-PII payload returned to the model."""
    return {
        "order_no": order.order_no,
        "status": order.status,
        "status_label": ORDER_STATUS_LABELS.get(order.status, order.status),
        "total_amount": round(order.total_amount or 0, 2),
        "paid_at": iso(order.paid_at),
        "delivered_at": iso(order.delivered_at),
        "recipient_phone": mask_phone(order.recipient_phone),
        "items": [
            {
                "item_id": item.id,
                "sku_id": item.sku_id,
                "product_name": item.product_name,
                "category": item.category,
                "quantity": item.quantity,
                "unit_price": round(item.unit_price or 0, 2),
                "item_status": item.item_status,
                "item_status_label": ITEM_STATUS_LABELS.get(item.item_status, item.item_status),
            }
            for item in order.items
        ],
    }


def order_display(order: Order) -> str:
    """Human-readable order summary. No address or full phone number."""
    item_text = "；".join(
        f"{item.product_name} ×{item.quantity}（{ITEM_STATUS_LABELS.get(item.item_status, item.item_status)}）"
        for item in order.items
    )
    delivered = f"，签收时间 {order.delivered_at:%Y-%m-%d %H:%M}" if order.delivered_at else ""
    return (
        f"订单 {order.order_no}，状态：{ORDER_STATUS_LABELS.get(order.status, order.status)}{delivered}，"
        f"合计 ¥{order.total_amount:.2f}。商品：{item_text or '无'}。"
    )
