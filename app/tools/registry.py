"""Tool registration and planning schemas."""
from __future__ import annotations

from app.tools.after_sales import PrepareReturnArgs, prepare_return
from app.tools.base import ToolSpec
from app.tools.inventory import CheckInventoryArgs, check_inventory
from app.tools.knowledge import KnowledgeSearchArgs, knowledge_search
from app.tools.logistics import QueryLogisticsArgs, query_logistics
from app.tools.orders import QueryOrderArgs, query_order
from app.tools.product import SearchProductArgs, search_product


class ToolRegistry:
    """In-process registry of tools exposed to the model."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate tool: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def all(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def planning_schemas(self) -> list[dict]:
        """Schemas for the tools the model may call.

        ``commit_return`` is intentionally absent: write execution is only
        reachable through the confirmation endpoint.
        """
        return [spec.planning_schema() for spec in self._specs.values()]


def build_default_registry() -> ToolRegistry:
    """Create the demo tool set.

    Read-only tools are safe to expose.  ``prepare_return`` is a write intent:
    it can create a PendingAction but cannot modify order data.
    """
    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="knowledge_search",
            label="检索售后政策",
            description="检索售后政策、退款规则、退货时效等知识库内容。回答政策类问题前先调用。",
            input_model=KnowledgeSearchArgs,
            handler=knowledge_search,
            risk="read",
            auth_scope="after_sales.read",
            timeout_seconds=30.0,
            retryable=True,
        )
    )
    registry.register(
        ToolSpec(
            name="query_order",
            label="查询订单",
            description="查询当前用户的订单状态、商品明细和金额。只能查询用户本人的订单。用户提到订单时先调用。",
            input_model=QueryOrderArgs,
            handler=query_order,
            risk="read",
            auth_scope="after_sales.read",
            timeout_seconds=5.0,
            retryable=True,
        )
    )
    registry.register(
        ToolSpec(
            name="query_logistics",
            label="查询物流",
            description="查询订单的物流公司、运单号和最新物流进度。",
            input_model=QueryLogisticsArgs,
            handler=query_logistics,
            risk="read",
            auth_scope="after_sales.read",
            timeout_seconds=5.0,
            retryable=True,
        )
    )
    registry.register(
        ToolSpec(
            name="check_inventory",
            label="查询库存",
            description="按 SKU 查询商品库存和可售数量。",
            input_model=CheckInventoryArgs,
            handler=check_inventory,
            risk="read",
            auth_scope="after_sales.read",
            timeout_seconds=5.0,
            retryable=True,
        )
    )
    registry.register(
        ToolSpec(
            name="search_product",
            label="查询商品",
            description=(
                "按商品名、SKU 或分类查询商品目录：返回产品说明、材质/尺码等属性、"
                "洗涤说明、发货与退换说明。用户询问商品规格、材质、洗涤方式、能否退换时优先调用。"
            ),
            input_model=SearchProductArgs,
            handler=search_product,
            risk="read",
            auth_scope="after_sales.read",
            timeout_seconds=5.0,
            retryable=True,
        )
    )
    registry.register(
        ToolSpec(
            name="prepare_return",
            label="准备退货",
            description=(
                "为订单商品创建退货待确认动作。该工具只做资格检查和退款金额计算，"
                "不会修改订单或退款；调用后系统会暂停并展示确认卡片，必须由用户确认后才会执行。"
            ),
            input_model=PrepareReturnArgs,
            handler=prepare_return,
            risk="write_intent",
            auth_scope="after_sales.write",
            timeout_seconds=10.0,
            retryable=False,
            requires_confirmation=True,
        )
    )
    return registry


_DEFAULT_REGISTRY: ToolRegistry | None = None


def get_default_registry() -> ToolRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = build_default_registry()
    return _DEFAULT_REGISTRY