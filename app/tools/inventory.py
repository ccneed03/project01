"""Legacy inventory lookup kept for wider commerce experiments.

The tool is intentionally not registered for the after-sales Agent: inventory
belongs to exchange workflows, which are outside the current MVP.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.agents.context import AgentContext
from app.agents.schemas import AgentState, ToolCall
from app.models.commerce import Inventory
from app.tools.base import ToolResult


class CheckInventoryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku_id: str = Field(description="商品 SKU 编号，例如 LAPTOP-001")


def check_inventory(
    args: CheckInventoryArgs,
    context: AgentContext,
    state: AgentState,
    call: ToolCall,
) -> ToolResult:
    sku_id = (args.sku_id or "").strip().upper()
    item = Inventory.query.filter_by(sku_id=sku_id).first()
    if not item:
        return ToolResult.error("NOT_FOUND", f"未找到商品 {sku_id} 的库存信息")

    available = max((item.stock or 0) - (item.reserved or 0), 0)
    display = (
        f"{item.product_name}（{item.sku_id}）当前库存 {item.stock} 件，"
        f"其中可售 {available} 件，仓库 {item.warehouse}。"
    )
    return ToolResult(
        ok=True,
        code="OK",
        data={
            "sku_id": item.sku_id,
            "product_name": item.product_name,
            "stock": item.stock,
            "reserved": item.reserved,
            "available": available,
            "warehouse": item.warehouse,
        },
        display_text=display,
    )
