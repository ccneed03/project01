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
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

POLICY_FILES = [
    PROJECT_ROOT / "data" / "policies" / "refund_policy.md",
    PROJECT_ROOT / "data" / "policies" / "electronics_after_sales_faq.md",
]
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


def check_vector_store(assistant, knowledge_base) -> None:
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
        rows = client.query(
            collection_name=Config.MILVUS_COLLECTION,
            filter=(
                f"knowledge_base_id == {knowledge_base.id} "
                'and doc_type == "document"'
            ),
            output_fields=["id"],
            limit=100,
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
            limit=100,
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
    from app.models.commerce import Order, Product, ReturnPolicy
    from app.models.document import Document
    from app.models.user import User
    from app.services.return_policy import ReturnPolicyEngine

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
            returnable = []
            for order in Order.query.filter_by(user_id=owner.id, status="delivered").all():
                items = [item for item in order.items if item.item_status == "normal"]
                decision = ReturnPolicyEngine.evaluate(
                    order=order,
                    items=items,
                    condition="unopened",
                    reason_code="no_longer_needed",
                )
                if decision.eligible:
                    returnable.append(order.order_no)
            report(
                "OK" if returnable else "FAIL",
                "Returnable orders",
                ", ".join(returnable) if returnable else "none",
                "python scripts/bootstrap_demo.py（默认 --reset，刷新签收时间）",
            )

        policies = ReturnPolicy.query.filter_by(active=True).all()
        versions = sorted({policy.policy_version for policy in policies})
        report(
            "OK" if policies else "FAIL",
            "Executable policy",
            f"{len(policies)} rules, version={', '.join(versions) if versions else 'none'}",
            "运行 python scripts/bootstrap_demo.py 重新加载 data/return_policies.json。",
        )
        policy_text = "\n".join(
            policy_file.read_text(encoding="utf-8")
            for policy_file in POLICY_FILES
            if policy_file.exists()
        )
        versions_in_text = [
            version for version in versions if f"政策版本：{version}" in policy_text
        ]
        report(
            "OK" if policies and len(versions_in_text) == len(versions) else "FAIL",
            "Policy text sync",
            (
                f"{len(versions_in_text)}/{len(versions)} version(s) present in Markdown"
                if versions
                else "no executable version"
            ),
            "同步维护 data/return_policies.json 与 data/policies/refund_policy.md 的政策版本。",
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
            "Orders",
            f"{Order.query.count()} order(s)",
        )

        policy_documents = []
        if knowledge_base:
            policy_documents = Document.query.filter(
                Document.knowledge_base_id == knowledge_base.id,
                Document.original_filename.in_(
                    [policy_file.name for policy_file in POLICY_FILES]
                ),
                Document.is_deleted.is_(False),
            ).all()
        missing_policy_files = [
            policy_file.name for policy_file in POLICY_FILES if not policy_file.exists()
        ]
        report(
            "OK" if not missing_policy_files else "FAIL",
            "Policy files",
            (
                f"{len(POLICY_FILES)} document(s)"
                if not missing_policy_files
                else f"missing: {', '.join(missing_policy_files)}"
            ),
            "确认 data/policies 下的退货政策和电子产品 FAQ 都存在。",
        )
        completed_policy_docs = [
            document
            for document in policy_documents
            if document.status == "completed"
        ]
        report(
            "OK" if len(completed_policy_docs) == len(POLICY_FILES) else "WARN",
            "Policy indexing",
            (
                f"{len(completed_policy_docs)}/{len(POLICY_FILES)} completed, "
                f"chunks={sum(document.chunk_count or 0 for document in completed_policy_docs)}"
            ),
            "Milvus/Ollama 可用时运行 bootstrap_demo.py；否则使用本地政策降级。",
        )
        report(
            "OK",
            "Demo sessions",
            f"{ChatSession.query.filter_by(assistant_id=assistant.id).count() if assistant else 0}",
        )

        check_vector_store(assistant, knowledge_base)
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
