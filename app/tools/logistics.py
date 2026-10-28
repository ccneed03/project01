"""Read-only logistics query tool."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.agents.context import AgentContext
from app.agents.schemas import AgentState, ToolCall
from app.tools.base import ToolResult
from app.tools.common import iso, resolve_order

SHIPMENT_STATUS_LABELS = {
    "created": "已创建",
    "in_transit": "运输中",
    "delivered": "已送达",
}


class QueryLogisticsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_no: str | None = Field(
        default=None,
        description="订单号。用户没有提供时可省略，此时使用当前会话中的订单。",
    )


def query_logistics(
    args: QueryLogisticsArgs,
    context: AgentContext,
    state: AgentState,
    call: ToolCall,
) -> ToolResult:
    order = resolve_order(args.order_no, context, state)
    if not order:
        if args.order_no:
            return ToolResult.error("NOT_FOUND", "未找到该订单，请确认订单号")
        return ToolResult.error("NEED_ORDER_NO", "请提供要查询物流的订单号")

    if not order.shipments:
        return ToolResult.error("NO_SHIPMENT", f"订单 {order.order_no} 暂无物流信息")

    shipments = [
        {
            "carrier": shipment.carrier,
            "tracking_no": shipment.tracking_no,
            "status": shipment.status,
            "status_label": SHIPMENT_STATUS_LABELS.get(shipment.status, shipment.status),
            "last_event": shipment.last_event,
            "shipped_at": iso(shipment.shipped_at),
            "delivered_at": iso(shipment.delivered_at),
        }
        for shipment in order.shipments
    ]
    latest = shipments[-1]
    display = (
        f"订单 {order.order_no} 物流：{latest['carrier']}，运单号 {latest['tracking_no']}，"
        f"状态：{latest['status_label']}，最新进度：{latest['last_event'] or '暂无更新'}。"
    )
    return ToolResult(
        ok=True,
        code="OK",
        data={"order_no": order.order_no, "shipments": shipments},
        display_text=display,
    )
