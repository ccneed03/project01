"""Check whether the local demo is ready to present.

Usage:
    python scripts/preflight.py
    python scripts/preflight.py --llm   # also call the configured LLM endpoint

Exit code is 1 when a blocking check fails.  Vector-store checks are warnings:
the agent has a local policy fallback and a product-catalog fallback, so the
core controlled-write demo still works without Milvus/Ollama.
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

POLICY_FILE = PROJECT_ROOT / "data" / "policies" / "refund_policy.md"
ASSISTANT_NAME = "售后助手"

results: list[tuple[str, str, str]] = []


def report(level: str, label: str, detail: str, fix: str = "") -> None:
    icon = {"OK": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[level]
    print(f"{icon} {label:<24} {detail}")
    if fix and level != "OK":
        print(f"       -> {fix}")
    results.append((level, label, detail))


def port_open(url: str, default_port: int, timeout: float = 1.2) -> bool:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or default_port
    sock = socket.socket()
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def check_vector_store(assistant, knowledge_base, policy_document) -> None:
    from app.config import Config

    milvus_up = port_open(Config.MILVUS_URI, 19530)
    ollama_up = port_open(Config.OLLAMA_BASE_URL, 11434)
    report(
        "OK" if milvus_up else "WARN",
        "Milvus",
        f"{Config.MILVUS_URI} {'reachable' if milvus_up else 'unreachable'}",
        "启动 Milvus 后重新运行 bootstrap；离线时 knowledge_search 使用本地政策降级。",
    )
    report(
        "OK" if ollama_up else "WARN",
        "Ollama",
        f"{Config.OLLAMA_BASE_URL} {'reachable' if ollama_up else 'unreachable'}",
        "启动 Ollama 并确认 embedding 模型已拉取。",
    )
    if not milvus_up or not knowledge_base:
        return

    try:
        from app.services.model_cache import get_milvus_client

        client = get_milvus_client(Config.MILVUS_URI, Config.MILVUS_TOKEN)
        if not client.has_collection(Config.MILVUS_COLLECTION):
            report(
                "WARN",
                "Milvus collection",
                f"{Config.MILVUS_COLLECTION} 不存在",
                "运行 python scripts/bootstrap_demo.py，或先索引任意文档以创建 collection。",
            )
            return

        client.load_collection(collection_name=Config.MILVUS_COLLECTION)
        if policy_document:
            rows = client.query(
                collection_name=Config.MILVUS_COLLECTION,
                filter=(
                    f"knowledge_base_id == {knowledge_base.id} "
                    f"and document_id == {policy_document.id}"
                ),
                output_fields=["id"],
                limit=1,
            )
            report(
                "OK" if rows else "WARN",
                "Policy vectors",
                f"{len(rows)} row(s)",
                "运行 python scripts/bootstrap_demo.py --skip-products 重新索引政策文档。",
            )

        rows = client.query(
            collection_name=Config.MILVUS_COLLECTION,
            filter=(
                f"knowledge_base_id == {knowledge_base.id} "
                'and doc_type == "product"'
            ),
            output_fields=["sku_id"],
            limit=1,
        )
        report(
            "OK" if rows else "WARN",
            "Product vectors",
            f"{len(rows)} row(s) in assistant KB",
            "运行 python scripts/index_products.py 把商品写入助手绑定的知识库。",
        )
    except Exception as exc:
        report(
            "WARN",
            "Milvus metadata query",
            f"{type(exc).__name__}: {str(exc)[:100]}",
            "老 collection 可能缺少 doc_type/category/sku_id 字段，需要重建后重新索引。",
        )


def check_llm() -> None:
    from app.config import Config

    key_set = bool(Config.LLM_API_KEY)
    report(
        "OK" if key_set else "FAIL",
        "LLM API key",
        f"{'set' if key_set else 'missing'} ({Config.LLM_BASE_URL}, {Config.LLM_MODEL})",
        "在 .env 中填写 D_API_KEY / D_BASE_URL / D_MODEL。",
    )


def check_llm_tools() -> None:
    script = PROJECT_ROOT / "scripts" / "smoke_llm_tools.py"
    print("       -> 调用 LLM 端点检查 tool_calls（可能产生少量费用）...")
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        report("OK", "LLM tool calling", "supported")
    else:
        detail = (completed.stderr or completed.stdout or "unknown error").strip().splitlines()
        report(
            "FAIL",
            "LLM tool calling",
            detail[-1][:160] if detail else "failed",
            "换一个支持 OpenAI tools 的模型/端点，再运行 python scripts/smoke_llm_tools.py。",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Check demo readiness")
    parser.add_argument("--llm", action="store_true", help="also call the LLM endpoint to verify tool_calls")
    args = parser.parse_args()

    print("=== 售后 Agent demo 自检 ===")
    check_llm()
    if args.llm:
        check_llm_tools()

    from app import create_app
    from app.config import Config
    from app.extensions import db
    from app.models.assistant import Assistant
    from app.models.chat import ChatSession
    from app.models.commerce import Inventory, Order, Product
    from app.models.document import Document
    from app.models.user import User
    from app.services.after_sales_service import NON_RETURNABLE_CATEGORIES

    app = create_app()
    with app.app_context():
        owner = User.query.filter_by(username="admin").first()
        report(
            "OK" if owner else "FAIL",
            "admin user",
            owner.username if owner else "missing",
            "运行 python scripts/bootstrap_demo.py 重新初始化演示数据。",
        )

        assistant = Assistant.query.filter_by(name=ASSISTANT_NAME, is_deleted=False).first()
        knowledge_base = assistant.knowledge_base if assistant else None
        report(
            "OK" if assistant else "FAIL",
            "Assistant",
            f"{assistant.name} (id={assistant.id})" if assistant else "missing",
            "运行 python scripts/bootstrap_demo.py。",
        )
        report(
            "OK" if knowledge_base else "FAIL",
            "Assistant KB binding",
            f"{knowledge_base.name} (id={knowledge_base.id})" if knowledge_base else "missing",
            "在后台编辑助手并绑定「售后政策」知识库。",
        )

        if owner:
            now = datetime.now()
            returnable = []
            for order in Order.query.filter_by(user_id=owner.id, status="delivered").all():
                if not order.delivered_at:
                    continue
                if now - order.delivered_at > timedelta(days=Config.AGENT_RETURN_WINDOW_DAYS):
                    continue
                items = [
                    item
                    for item in order.items
                    if item.item_status == "normal"
                    and item.category not in NON_RETURNABLE_CATEGORIES
                ]
                if items:
                    returnable.append(order.order_no)
            report(
                "OK" if returnable else "FAIL",
                f"Returnable orders ({Config.AGENT_RETURN_WINDOW_DAYS}d)",
                ", ".join(returnable) if returnable else "none",
                "python scripts/bootstrap_demo.py（默认 --reset，刷新签收时间）",
            )

        products = Product.query.count()
        report(
            "OK" if products else "FAIL",
            "Product catalog",
            f"{products} product(s)",
            "运行 python scripts/bootstrap_demo.py。",
        )
        report(
            "OK",
            "Inventory / orders",
            f"{Inventory.query.count()} inventory / {Order.query.count()} orders",
        )

        policy_document = None
        if knowledge_base:
            policy_document = Document.query.filter_by(
                knowledge_base_id=knowledge_base.id,
                original_filename=POLICY_FILE.name,
                is_deleted=False,
            ).first()
        report(
            "OK" if POLICY_FILE.exists() else "FAIL",
            "Policy file",
            str(POLICY_FILE.relative_to(PROJECT_ROOT)) if POLICY_FILE.exists() else "missing",
            "确认 data/policies/refund_policy.md 存在。",
        )
        report(
            "OK" if policy_document and policy_document.status == "completed" else "WARN",
            "Policy indexing",
            (
                f"document {policy_document.id}, status={policy_document.status}, "
                f"chunks={policy_document.chunk_count or 0}"
                if policy_document
                else "not uploaded"
            ),
            "Milvus/Ollama 可用时运行 bootstrap_demo.py；否则使用本地政策降级。",
        )
        report(
            "OK",
            "Demo sessions",
            f"{ChatSession.query.filter_by(assistant_id=assistant.id).count() if assistant else 0}",
        )

        check_vector_store(assistant, knowledge_base, policy_document)
        db.session.remove()

    failures = sum(1 for level, _, _ in results if level == "FAIL")
    warnings = sum(1 for level, _, _ in results if level == "WARN")
    print("")
    if failures:
        print(f"结果：{failures} 项失败，{warnings} 项警告。先修复 FAIL 再演示。")
        return 1
    print(f"结果：通过（{warnings} 项警告）。可以启动 python run.py。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
