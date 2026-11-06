"""Write-intent tool: prepare (never execute) a return."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agents.context import AgentContext
from app.agents.schemas import AgentState, ToolCall
from app.services.after_sales_service import AfterSalesError, AfterSalesService
from app.tools.base import ToolResult
from app.tools.common import resolve_order


class PrepareReturnArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_no: str = Field(description="要退货的订单号")
    item_ids: list[int] = Field(min_length=1, description="要退货的订单商品 item_id 列表")
    reason: str = Field(default="", max_length=500, description="用户给出的退货原因")
    reason_code: Literal[
        "no_longer_needed",
        "size_or_fit",
        "damaged_on_arrival",
        "defective_quality",
        "wrong_item",
        "other",
    ] = Field(description="归一化后的退货原因；无法判断时先向用户澄清")
    condition: Literal["unopened", "opened", "activated", "used", "damaged"] = Field(
        description="商品当前状态；用户未说明是否拆封、激活或使用时先询问，不得猜测"
    )


def prepare_return(
    args: PrepareReturnArgs,
    context: AgentContext,
    state: AgentState,
    call: ToolCall,
) -> ToolResult:
    order = resolve_order(args.order_no, context, state)
    if not order:
        return ToolResult.error("NOT_FOUND", "未找到该订单，请确认订单号")

    try:
        pending = AfterSalesService.prepare_return(
            order=order,
            item_ids=args.item_ids,
            reason=args.reason,
            reason_code=args.reason_code,
            condition=args.condition,
            user_id=context.user_id,
            session_id=context.session_id,
            run_id=context.run_id,
            tool_call_id=call.id,
        )
    except AfterSalesError as exc:
        return ToolResult.error(exc.code, exc.message)
    except Exception:
        return ToolResult.error("TOOL_ERROR", "售后预检查失败，请稍后再试")

    display = dict(pending.display_json or {})
    display["action_id"] = pending.id
    display["confirmation_token"] = AfterSalesService.confirmation_token(pending)
    return ToolResult(
        ok=True,
        code="CONFIRMATION_REQUIRED",
        data={
            "action_id": pending.id,
            "summary": display.get("summary"),
            "expires_at": display.get("expires_at"),
            "policy_version": display.get("policy_version"),
            "decision_code": display.get("decision_code"),
            "requires_review": display.get("requires_review"),
        },
        display_text=display.get("summary") or "退货申请待确认",
        pending_action_id=pending.id,
        pending_action=display,
    )
