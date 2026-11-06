"""Read-only product catalog query tool.

The product catalog is authoritative relational data (``Product`` table). Its
description / care / delivery / return texts are natural-language content that
may also be indexed into the vector store via the standard Milvus pipeline; the
tool itself is DB-backed so it works and is testable without Milvus.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import or_

from app.agents.context import AgentContext
from app.agents.schemas import AgentState, ToolCall
from app.extensions import db
from app.models.commerce import Product
from app.models.knowledge_base import KnowledgeBase
from app.tools.base import ToolResult

RESULT_LIMIT = 5


class SearchProductArgs(BaseModel):
    """At least one of ``query`` / ``sku_id`` / ``category`` is expected."""

    model_config = ConfigDict(extra="forbid")

    query: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description='产品名或说明关键字，例如 "笔记本"、"降噪"、"4K"',
    )
    sku_id: str | None = Field(
        default=None,
        description="商品 SKU 编号，例如 LAPTOP-001。精确匹配，优先级最高。",
    )
    category: str | None = Field(
        default=None,
        max_length=64,
        description="商品分类，例如 笔记本 / 手机 / 耳机 / 外设 / 定制 / 虚拟商品，可配合 query 一起使用。",
    )


def _product_view(product: Product) -> dict[str, Any]:
    """Safe, structured product payload returned to the model."""
    return {
        "sku_id": product.sku_id,
        "product_name": product.product_name,
        "category": product.category,
        "brand": product.brand,
        "description": product.description,
        "details": product.details or {},
        "care_info": product.care_info,
        "delivery_note": product.delivery_note,
        "return_note": product.return_note,
    }


def _quote_attr(value: str) -> str:
    """Escape a string literal for a Milvus filter expression."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _product_filter_expr(category: str | None, sku_id: str | None) -> str:
    """Build the scalar metadata filter narrowing hits to the product catalog."""
    expr = f'doc_type == "product"'
    if category:
        expr += f" and category == {_quote_attr(category)}"
    if sku_id:
        expr += f" and sku_id == {_quote_attr(sku_id)}"
    return expr


def _display(products: list[Product]) -> str:
    if len(products) == 1:
        product = products[0]
        preview = (product.description or "").strip()
        if len(preview) > 60:
            preview = preview[:60] + "…"
        return (
            f"{product.product_name}（{product.sku_id}），分类：{product.category}，"
            f"品牌：{product.brand or '未知'}。说明：{preview or '暂无'}。"
        )
    names = "；".join(f"{p.product_name}（{p.sku_id}）" for p in products)
    return f"找到 {len(products)} 个相关商品：{names}。"


def _search_semantic(
    args: SearchProductArgs,
    context: AgentContext,
    sku_id: str,
    category: str,
) -> list[Product] | None:
    """Semantic retrieval over the product catalog indexed in Milvus.

    Returns a product list, or ``None`` to signal the caller to fall back to
    the authoritative DB catalog (e.g. when no KB is bound, Milvus is down, or
    the query is an exact keyword the semantic layer cannot satisfy).
    """
    if not context.knowledge_base_id:
        return None
    knowledge_base = db.session.get(KnowledgeBase, context.knowledge_base_id)
    if not knowledge_base or knowledge_base.is_deleted:
        return None

    from app.services.retrieval_service import RetrievalService

    config = context.retrieval_config or {}
    try:
        retrieval = RetrievalService(knowledge_base)
        filter_expr = _product_filter_expr(category, sku_id or None)
        hits = retrieval.search(
            query=(args.query or args.category or "").strip() or "",
            method=config.get("retrieval_mode") or "hybrid",
            top_k=RESULT_LIMIT,
            enable_rerank=config.get("enable_rerank", True),
            rerank_model=config.get("rerank_model"),
            score_threshold=config.get("score_threshold"),
            filter_expr=filter_expr,
            strict=True,
        )
    except Exception:  # Milvus / embedding / reranker unavailable
        return None

    if not hits:
        return None

    # Map hit sku_id -> authoritative product so we never trust vector content.
    products: list[Product] = []
    seen: set[str] = set()
    for hit in hits:
        hit_sku = (hit.get("sku_id") or "").strip()
        if not hit_sku or hit_sku in seen:
            continue
        product = Product.query.filter_by(sku_id=hit_sku).first()
        if product:
            seen.add(hit_sku)
            products.append(product)
    return products or None


def search_product(
    args: SearchProductArgs,
    context: AgentContext,
    state: AgentState,
    call: ToolCall,
) -> ToolResult:
    sku_id = (args.sku_id or "").strip().upper()
    query = (args.query or "").strip()
    category = (args.category or "").strip()

    if not sku_id and not query and not category:
        return ToolResult.error("NEED_QUERY", "请提供要查询的商品名、SKU 或分类")

    # Exact SKU is authoritative and cheap; never go to the vector store for it.
    if sku_id:
        product = Product.query.filter_by(sku_id=sku_id).first()
        if not product:
            return ToolResult.error(
                "NOT_FOUND", f"未找到商品 {sku_id}，请确认 SKU 是否正确"
            )
        products = [product]
    else:
        products = _search_semantic(args, context, sku_id, category)
        if not products:
            # Fall back to the authoritative DB catalog (rule-based keyword match).
            record = Product.query
            if category:
                record = record.filter(Product.category == category)
            if query:
                keyword = f"%{query}%"
                record = record.filter(
                    or_(
                        Product.product_name.ilike(keyword),
                        Product.description.ilike(keyword),
                        Product.care_info.ilike(keyword),
                    )
                )
            products = record.order_by(Product.id).limit(RESULT_LIMIT).all()
            if not products:
                return ToolResult.error("NO_RESULT", "没有找到匹配的商品，请换一个关键字试试")

    data = [_product_view(product) for product in products]
    return ToolResult(
        ok=True,
        code="OK",
        data={"total": len(data), "products": data},
        display_text=_display(products),
    )
