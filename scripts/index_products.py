"""Index the authoritative product catalog into Milvus for semantic retrieval.

The ``search_product`` tool first tries a vector/hybrid semantic search of the
product texts and falls back to a rule-based DB catalog lookup when the vector
store is unavailable. Semantic search only finds a product once that product's
text has been indexed here.

Standalone pieces (no app running):

- Collection schema is ensured by ``DocumentService._ensure_collection`` and now
  includes three scalar metadata columns: ``doc_type``, ``category``, ``sku_id``.
- Each product is stored as one row tagged ``doc_type == "product"`` with its
  ``category`` and ``sku_id``, so ``search_product`` can narrow semantic hits
  with a Milvus scalar metadata filter expression.

By default the products are written into the knowledge base bound to the demo
assistant ("售后助手"), because ``search_product`` only searches the assistant's
own knowledge base.  Passing ``--kb-name`` targets a specific knowledge base
instead.

Usage:
    python scripts/index_products.py                      # assistant's KB
    python scripts/index_products.py --kb-name 售后政策   # explicit KB
    python scripts/index_products.py --reset              # drop product rows first

Requires Milvus and Ollama to be reachable (see app.config.Config).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app  # noqa: E402
from app.config import Config  # noqa: E402
from app.extensions import db  # noqa: E402
from app.models.assistant import Assistant  # noqa: E402
from app.models.commerce import Product  # noqa: E402
from app.models.knowledge_base import KnowledgeBase  # noqa: E402

DEFAULT_ASSISTANT_NAME = "售后助手"
FALLBACK_KB_NAME = "产品库"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index products into Milvus")
    parser.add_argument(
        "--kb-name",
        default=None,
        help="knowledge base that owns the product rows (default: the assistant's KB)",
    )
    parser.add_argument(
        "--assistant",
        default=DEFAULT_ASSISTANT_NAME,
        help="assistant whose bound knowledge base is used by default",
    )
    parser.add_argument("--reset", action="store_true", help="delete existing product-type rows first")
    return parser.parse_args()


def _get_or_create_kb(name: str) -> KnowledgeBase:
    knowledge_base = KnowledgeBase.query.filter_by(name=name, is_deleted=False).first()
    if knowledge_base:
        return knowledge_base
    knowledge_base = KnowledgeBase(
        name=name,
        description="商品说明向量库（产品语义检索）",
        default_embedding_model=Config.OLLAMA_EMBEDDING_MODEL,
    )
    db.session.add(knowledge_base)
    db.session.commit()
    return knowledge_base


def resolve_knowledge_base(
    *,
    kb_name: str | None = None,
    assistant_name: str = DEFAULT_ASSISTANT_NAME,
) -> tuple[KnowledgeBase, str]:
    """Resolve where product rows belong.

    Priority: explicit ``--kb-name`` > the assistant's bound KB > fallback
    "产品库".  The assistant binding matters because ``search_product`` uses
    ``context.knowledge_base_id`` for its Milvus filter.
    """
    if kb_name:
        return _get_or_create_kb(kb_name), f"explicit --kb-name {kb_name}"

    assistant = Assistant.query.filter_by(name=assistant_name, is_deleted=False).first()
    if assistant:
        binding = assistant.primary_knowledge_base_config
        knowledge_base = binding.knowledge_base if binding else None
        if knowledge_base and not knowledge_base.is_deleted:
            return knowledge_base, f"assistant {assistant.name}"
        print(
            f"[warn] 助手「{assistant_name}」未绑定知识库，"
            f"商品索引将写入「{FALLBACK_KB_NAME}」，search_product 不会命中。"
        )
    else:
        print(
            f"[warn] 未找到助手「{assistant_name}」，"
            f"商品索引将写入「{FALLBACK_KB_NAME}」。"
        )
    return _get_or_create_kb(FALLBACK_KB_NAME), f"fallback {FALLBACK_KB_NAME}"


def _build_product_text(product: Product) -> str:
    """Flatten a product into the searchable natural-language text."""
    parts = [
        product.product_name,
        f"分类：{product.category or '—'}",
        f"品牌：{product.brand or '—'}",
    ]
    if product.description:
        parts.append(product.description)
    if product.details:
        details = "，".join(f"{k}：{v}" for k, v in product.details.items())
        parts.append(f"属性：{details}")
    if product.care_info:
        parts.append(f"洗涤保养：{product.care_info}")
    if product.delivery_note:
        parts.append(f"发货说明：{product.delivery_note}")
    if product.return_note:
        parts.append(f"退换说明：{product.return_note}")
    return "\n".join(p for p in parts if p)


def index_products_into_kb(
    knowledge_base: KnowledgeBase,
    *,
    reset: bool = False,
) -> int:
    """Embed and insert all products into ``knowledge_base``.

    Runs synchronously and raises on Milvus/Ollama failure so callers (CLI or
    bootstrap script) can decide whether to degrade gracefully.
    """
    from app.services.document_service import DocumentService
    from app.services.model_cache import get_milvus_client
    from app.services.ollama_service import OllamaService

    products = Product.query.order_by(Product.id).all()
    if not products:
        raise RuntimeError("商品目录为空，请先运行 python scripts/seed_demo.py")

    texts = [_build_product_text(product) for product in products]
    embedding_model = (
        knowledge_base.default_embedding_model or Config.OLLAMA_EMBEDDING_MODEL
    ).strip()

    client = get_milvus_client(Config.MILVUS_URI, Config.MILVUS_TOKEN)
    DocumentService._ensure_collection(client)
    if reset:
        client.delete(
            collection_name=Config.MILVUS_COLLECTION,
            filter=f'knowledge_base_id == {knowledge_base.id} and doc_type == "product"',
        )

    vectors = OllamaService.embed_documents(texts, model=embedding_model)
    data = [
        {
            "knowledge_base_id": knowledge_base.id,
            # Negative ids keep product rows out of the Document id space.
            # DocumentService._delete_from_milvus() deletes by
            # ``document_id == doc.id``; sharing ids would let a policy
            # reindex wipe product rows for the same numeric id.
            "document_id": -product.id,
            "chunk_index": 0,
            "text": text,
            "dense": vector,
            "doc_type": "product",
            "category": product.category or "",
            "sku_id": (product.sku_id or "").upper(),
        }
        for product, text, vector in zip(products, texts, vectors)
    ]

    result = client.insert(collection_name=Config.MILVUS_COLLECTION, data=data)
    client.flush(collection_name=Config.MILVUS_COLLECTION)
    client.load_collection(collection_name=Config.MILVUS_COLLECTION)
    return len(result["ids"]) if isinstance(result, dict) else len(result)


def main() -> int:
    args = parse_args()
    app = create_app()
    with app.app_context():
        knowledge_base, source = resolve_knowledge_base(
            kb_name=args.kb_name,
            assistant_name=args.assistant,
        )
        print(f"索引目标：知识库「{knowledge_base.name}」（id={knowledge_base.id}，{source}）")

        try:
            count = index_products_into_kb(knowledge_base, reset=args.reset)
        except Exception as exc:  # Milvus / Ollama unavailable
            print(
                f"[{type(exc).__name__}] 索引失败：{exc}\n"
                "请确认 Milvus 与 Ollama 已启动。语义检索暂不可用，"
                "search_product 仍会通过数据库目录提供查询。"
            )
            return 1

        print(
            f"已索引 {count} 条商品 -> 知识库「{knowledge_base.name}」"
            "（doc_type=product，带 category/sku_id 标量字段）"
        )
        print("提示：语义检索会过滤 doc_type == \"product\"，并回落到商品目录表兜底。")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
