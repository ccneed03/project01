"""Read-only order query tool."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.agents.context import AgentContext
from app.agents.schemas import AgentState, ToolCall
from app.tools.base import ToolResult
from app.tools.common import order_data, order_display, resolve_order


class QueryOrderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_no: str | None = Field(
        default=None,
        description="订单号。用户没有提供时可省略，此时使用当前会话中的订单。",
    )


def query_order(
    args: QueryOrderArgs,
    context: AgentContext,
    state: AgentState,
    call: ToolCall,
) -> ToolResult:
    order = resolve_order(args.order_no, context, state)
    if not order:
        if args.order_no:
            return ToolResult.error("NOT_FOUND", "未找到该订单，请确认订单号")
        return ToolResult.error("NEED_ORDER_NO", "请提供要查询的订单号")

    return ToolResult(
        ok=True,
        code="OK",
        data=order_data(order),
        display_text=order_display(order),
    )
