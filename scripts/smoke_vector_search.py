"""Smoke-test product and policy retrieval against the demo vector store."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app  # noqa: E402
from app.models.assistant import Assistant  # noqa: E402
from app.services.retrieval_service import RetrievalService  # noqa: E402


def main() -> int:
    app = create_app()
    with app.app_context():
        assistant = Assistant.query.filter_by(
            name="售后助手", is_deleted=False
        ).first()
        if not assistant or not assistant.knowledge_base:
            print("[FAIL] 售后助手或绑定知识库不存在")
            return 1

        retrieval = RetrievalService(assistant.knowledge_base)
        tries = [
            (
                "商品语义检索",
                "适合通勤的降噪耳机",
                'doc_type == "product"',
            ),
            ("政策检索", "已经激活的笔记本还能退货吗", None),
        ]
        ok = True
        for label, query, filter_expr in tries:
            try:
                hits = retrieval.search(
                    query,
                    method="hybrid",
                    top_k=5,
                    enable_rerank=False,
                    filter_expr=filter_expr,
                    strict=True,
                )
            except Exception as exc:
                print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
                ok = False
                continue

            if not hits:
                print(f"[FAIL] {label}: no hits")
                ok = False
                continue
            first = hits[0]
            identity = first.get("sku_id") or first.get("doc_type") or first.get("id")
            text = (first.get("text") or "").replace("\n", " ")[:80]
            print(
                f"[ OK ] {label}: id={identity}, "
                f"score={float(first.get('score', 0)):.5f}, text={text}"
            )

        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
