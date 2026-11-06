"""Knowledge-base retrieval wrapped as an agent tool.

The vector path is the primary source.  When Milvus/Ollama is unavailable (a
common state for the local demo), the tool degrades to a small local markdown
search over ``data/policies/refund_policy.md`` so policy-grounded answers still
work.  The fallback is explicitly labelled in the tool result and sources.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agents.context import AgentContext
from app.agents.schemas import AgentState, ToolCall
from app.extensions import db
from app.models.knowledge_base import KnowledgeBase
from app.tools.base import ToolResult

POLICY_DIR = Path(__file__).resolve().parents[2] / "data" / "policies"
POLICY_FILES = [
    POLICY_DIR / "refund_policy.md",
    POLICY_DIR / "electronics_after_sales_faq.md",
]

# Small, explicit keyword table for the demo fallback.  This is intentionally
# not a production search engine; it only needs to retrieve the right section
# of the demo policy file when the vector store is offline.
POLICY_KEYWORDS: dict[str, float] = {
    "退货": 6,
    "退款": 5,
    "退换": 5,
    "无理由": 5,
    "七天": 4,
    "7天": 4,
    "7 天": 4,
    "签收": 3,
    "定制": 4,
    "虚拟": 4,
    "运费": 3,
    "到账": 3,
    "超期": 4,
    "质量": 4,
    "流程": 3,
    "条件": 2,
    "不支持": 3,
    "时效": 3,
    "能退": 3,
    "退吗": 2,
    "退货期": 4,
    "电子": 4,
    "激活": 5,
    "序列号": 3,
    "软件": 4,
    "授权": 4,
    "激活码": 5,
    "保修": 3,
    "配件": 3,
    "拆封": 3,
    "退": 1,
}


class KnowledgeSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=512, description="要检索的售后政策或业务问题")
    top_k: int = Field(default=5, ge=1, le=8, description="返回的相关片段数量")


def _policy_sections() -> list[dict[str, Any]]:
    """Read all demo policy files and split them by level-2 headings."""
    sections: list[dict[str, Any]] = []
    for policy_file in POLICY_FILES:
        if not policy_file.exists():
            continue
        text = policy_file.read_text(encoding="utf-8")
        document = {
            "filename": policy_file.name,
            "original_filename": policy_file.name,
            "source": "local_fallback",
        }
        matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", text))
        if not matches:
            stripped = text.strip()
            if stripped:
                sections.append(
                    {
                        "title": "电子产品售后政策",
                        "text": stripped,
                        "document": document,
                    }
                )
            continue

        for position, match in enumerate(matches):
            start = match.start()
            end = (
                matches[position + 1].start()
                if position + 1 < len(matches)
                else len(text)
            )
            body = text[start:end].strip()
            if body:
                sections.append(
                    {
                        "title": match.group(1).strip(),
                        "text": body,
                        "document": document,
                    }
                )
    return sections


def _compact(value: str) -> str:
    return re.sub(r"[\s\W_]+", "", value.lower(), flags=re.UNICODE)


def _query_bigrams(query: str) -> set[str]:
    compact = _compact(query)
    return {compact[index : index + 2] for index in range(len(compact) - 1)}


def _score_section(query: str, bigrams: set[str], section: dict[str, Any]) -> float:
    title = section["title"]
    text = section["text"]
    compact_text = _compact(text)
    score = 0.0

    for keyword, weight in POLICY_KEYWORDS.items():
        if keyword not in query:
            continue
        occurrences = text.count(keyword)
        if occurrences:
            score += weight * min(occurrences, 3)
        if keyword in title:
            score += weight * 2

    if bigrams:
        overlap = sum(1 for bigram in bigrams if bigram in compact_text)
        score += min(overlap * 0.4, 6.0)

    if section["title"] in query:
        score += 3.0
    return score


def _search_local_policy(query: str, top_k: int) -> list[dict[str, Any]]:
    """Keyword/bigram search over the demo policy file.

    Returns hits shaped like ``RetrievalService.search`` results so the rest of
    the tool and the frontend can handle both paths identically.
    """
    sections = _policy_sections()
    if not sections:
        return []

    bigrams = _query_bigrams(query)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for position, section in enumerate(sections):
        score = _score_section(query, bigrams, section)
        if score > 0:
            scored.append((score, position, section))

    if not scored:
        return []

    scored.sort(key=lambda item: item[0], reverse=True)
    max_score = scored[0][0] or 1.0
    return [
        {
            "text": section["text"],
            "document": section["document"],
            "chunk_index": position,
            "score": round(score / max_score, 4),
        }
        for score, position, section in scored[:top_k]
    ]


def _format_hits(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
    return snippets, sources


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
    retrieval_error: Exception | None = None
    results: list[dict[str, Any]] = []
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
        retrieval_error = exc

    snippets, sources = _format_hits(results)
    degraded = False
    if not snippets:
        # The demo should still be able to quote the refund policy when the
        # vector stack is offline or the document has not been indexed yet.
        fallback = _search_local_policy(args.query, args.top_k)
        if fallback:
            snippets, sources = _format_hits(fallback)
            degraded = True

    if not snippets:
        if retrieval_error is not None:
            return ToolResult.error(
                "RETRIEVAL_UNAVAILABLE",
                "知识库暂时不可用，请稍后再试",
                data={"error": str(retrieval_error)[:300]},
                retryable=True,
            )
        return ToolResult.error("NO_RESULT", "知识库中没有找到相关政策说明")

    preview = "\n\n".join(
        f"【{index + 1}】{snippet['text'][:400]}"
        for index, snippet in enumerate(snippets[:2])
    )
    prefix = "知识库检索结果（本地政策降级）：" if degraded else "知识库检索结果："
    data: dict[str, Any] = {"query": args.query, "snippets": snippets}
    if degraded:
        data["degraded"] = True
        data["degraded_reason"] = (
            "vector retrieval unavailable; used local refund_policy.md"
            if retrieval_error is not None
            else "vector retrieval returned no hits; used local refund_policy.md"
        )
    return ToolResult(
        ok=True,
        code="OK_DEGRADED" if degraded else "OK",
        data=data,
        sources=sources,
        display_text=f"{prefix}\n{preview}",
    )
