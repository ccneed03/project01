"""Knowledge-base retrieval wrapped as an agent tool."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agents.context import AgentContext
from app.agents.schemas import AgentState, ToolCall
from app.extensions import db
from app.models.knowledge_base import KnowledgeBase
from app.tools.base import ToolResult


class KnowledgeSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=512, description="要检索的售后政策或业务问题")
    top_k: int = Field(default=5, ge=1, le=8, description="返回的相关片段数量")


def knowledge_search(
    args: KnowledgeSearchArgs,
    context: AgentContext,
    state: AgentState,
    call: ToolCall,
) -> ToolResult:
    if not context.knowledge_base_id:
        return ToolResult.error("NO_KNOWLEDGE_BASE", "当前应用未绑定知识库，暂时无法查询售后政策")

    knowledge_base = db.session.get(KnowledgeBase, context.knowledge_base_id)
    if not knowledge_base or knowledge_base.is_deleted:
        return ToolResult.error("NO_KNOWLEDGE_BASE", "知识库不存在或已删除")

    # Lazy import keeps model/tool unit tests independent from Milvus.
    from app.services.retrieval_service import RetrievalService

    config = context.retrieval_config or {}
    try:
        retrieval = RetrievalService(knowledge_base)
        results = retrieval.search(
            query=args.query,
            method=config.get("retrieval_mode") or "hybrid",
            top_k=args.top_k,
            enable_rerank=config.get("enable_rerank", True),
            rerank_model=config.get("rerank_model"),
            score_threshold=config.get("score_threshold"),
            strict=True,
        )
    except Exception as exc:  # Milvus / embedding / reranker failure
        return ToolResult.error(
            "RETRIEVAL_UNAVAILABLE",
            "知识库暂时不可用，请稍后再试",
            data={"error": str(exc)[:300]},
            retryable=True,
        )

    if not results:
        return ToolResult.error("NO_RESULT", "知识库中没有找到相关政策说明")

    snippets: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for hit in results:
        text = (hit.get("text") or "").strip()
        if not text:
            continue
        document = hit.get("document") or {}
        snippets.append(
            {
                "text": text[:800],
                "document": document.get("original_filename") or document.get("filename"),
                "chunk_index": hit.get("chunk_index"),
                "score": hit.get("score"),
            }
        )
        sources.append(
            {
                "text": text[:800],
                "score": hit.get("score"),
                "chunk_index": hit.get("chunk_index"),
                "document": document,
            }
        )

    if not snippets:
        return ToolResult.error("NO_RESULT", "知识库中没有找到相关政策说明")

    preview = "\n\n".join(
        f"【{index + 1}】{snippet['text'][:400]}"
        for index, snippet in enumerate(snippets[:2])
    )
    return ToolResult(
        ok=True,
        code="OK",
        data={"query": args.query, "snippets": snippets},
        sources=sources,
        display_text=f"知识库检索结果：\n{preview}",
    )
