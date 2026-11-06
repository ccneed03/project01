"""One-command demo bootstrap.

What it does:

1. Reseeds the demo commerce data with fresh relative dates (orders are
   otherwise left untouched by the idempotent seed script and soon fall
   outside the 7-day return window).
2. Clears old demo chat sessions / agent runs so repeated demos start clean.
3. Optionally indexes ``data/policies/refund_policy.md`` and the product
   catalog into the assistant's knowledge base when Milvus + Ollama are up.
   When they are down, the agent still works: ``knowledge_search`` falls back
   to the local policy file.

Usage:
    python scripts/bootstrap_demo.py
    python scripts/bootstrap_demo.py --no-reset
    python scripts/bootstrap_demo.py --skip-index
    python scripts/bootstrap_demo.py --skip-products

Run it while the Flask app is stopped; SQLite writes from two processes can
otherwise block each other.
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

POLICY_FILE = PROJECT_ROOT / "data" / "policies" / "refund_policy.md"
FAQ_FILE = PROJECT_ROOT / "data" / "policies" / "electronics_after_sales_faq.md"
POLICY_FILES = [POLICY_FILE, FAQ_FILE]
DEFAULT_ASSISTANT_NAME = "售后助手"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a repeatable demo environment")
    parser.add_argument("--username", default="admin", help="user that owns the demo orders")
    parser.add_argument("--other-username", default="demo_other", help="user used for ownership tests")
    parser.add_argument("--no-reset", action="store_true", help="do not recreate demo orders/dates")
    parser.add_argument("--no-purge-chat", action="store_true", help="keep old demo chat sessions/runs")
    parser.add_argument("--skip-index", action="store_true", help="skip all Milvus indexing")
    parser.add_argument("--skip-policy", action="store_true", help="skip policy document indexing")
    parser.add_argument("--skip-products", action="store_true", help="skip product semantic indexing")
    return parser.parse_args()


def port_open(url: str, default_port: int, timeout: float = 1.0) -> bool:
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


def vector_stack_ready() -> bool:
    from app.config import Config

    return port_open(Config.MILVUS_URI, 19530) and port_open(Config.OLLAMA_BASE_URL, 11434)


def run_seed(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "seed_demo.py"),
        "--username",
        args.username,
        "--other-username",
        args.other_username,
    ]
    if not args.no_reset:
        command.append("--reset")
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


def purge_demo_chat(assistant, owner, other) -> None:
    """Remove old sessions/runs for the demo assistant.

    ``seed_demo.py --reset`` removes pending actions and return requests for
    the recreated orders, but leaves chat history and audit rows behind.  They
    reference deleted order ids, so a clean demo should not show them.
    """
    from sqlalchemy import and_, or_

    from app.extensions import db
    from app.models.agent import AgentRun, AgentStep, PendingAction
    from app.models.chat import ChatMessage, ChatSession

    conditions = [ChatSession.assistant_id == assistant.id]
    knowledge_base = assistant.knowledge_base
    if knowledge_base:
        conditions.append(
            and_(
                ChatSession.user_id.in_([owner.id, other.id]),
                ChatSession.knowledge_base_id == knowledge_base.id,
            )
        )

    sessions = ChatSession.query.filter(or_(*conditions)).all()
    session_ids = [session.id for session in sessions]
    if not session_ids:
        print("[chat] 没有需要清理的演示会话")
        return

    run_ids = [
        row.run_id
        for row in AgentRun.query.filter(AgentRun.session_id.in_(session_ids)).all()
    ]
    PendingAction.query.filter(PendingAction.session_id.in_(session_ids)).delete(
        synchronize_session=False
    )
    if run_ids:
        AgentStep.query.filter(AgentStep.run_id.in_(run_ids)).delete(
            synchronize_session=False
        )
    AgentRun.query.filter(AgentRun.session_id.in_(session_ids)).delete(
        synchronize_session=False
    )
    ChatMessage.query.filter(ChatMessage.session_id.in_(session_ids)).delete(
        synchronize_session=False
    )
    ChatSession.query.filter(ChatSession.id.in_(session_ids)).delete(
        synchronize_session=False
    )
    db.session.commit()
    print(
        f"[chat] 已清理 {len(session_ids)} 个演示会话、"
        f"{len(run_ids)} 条 agent run"
    )


def index_policy_file(assistant, owner, policy_file: Path) -> None:
    """Upload and index one policy document into the assistant's KB."""
    from werkzeug.datastructures import FileStorage

    from app.config import Config
    from app.extensions import db
    from app.models.document import Document
    from app.services.document_index_service import DocumentIndexService
    from app.services.document_service import DocumentService
    from app.services.model_catalog_service import ModelCatalogService

    knowledge_base = assistant.knowledge_base
    if not knowledge_base:
        print("[policy] 助手未绑定知识库，跳过向量索引（本地降级仍可用）")
        return
    if not policy_file.exists():
        print(f"[policy] 未找到 {policy_file}，跳过向量索引")
        return

    embedding_model = (
        knowledge_base.default_embedding_model or Config.OLLAMA_EMBEDDING_MODEL
    ).strip()
    try:
        available = bool(ModelCatalogService.resolve_embedding_model(embedding_model))
    except Exception as exc:
        print(f"[policy] Ollama 不可用（{type(exc).__name__}），跳过向量索引")
        print("          knowledge_search 将使用本地政策文件降级检索。")
        return
    if not available:
        print(f"[policy] Ollama 中没有 embedding 模型「{embedding_model}」，跳过向量索引")
        print("          knowledge_search 将使用本地政策文件降级检索。")
        return

    document = Document.query.filter_by(
        knowledge_base_id=knowledge_base.id,
        original_filename=policy_file.name,
        is_deleted=False,
    ).first()
    if document and not Path(document.file_path).exists():
        document.is_deleted = True
        db.session.commit()
        document = None

    if document and document.status == "completed" and (document.chunk_count or 0) > 0:
        print(f"[policy] 已索引（{document.chunk_count} chunks），跳过")
        return

    if not document:
        file_storage = FileStorage(
            stream=BytesIO(policy_file.read_bytes()),
            filename=policy_file.name,
            content_type="text/markdown",
        )
        ok, message, document = DocumentService.create_document(
            knowledge_base.id, file_storage, owner.id
        )
        if not ok or not document:
            print(f"[policy] 上传失败：{message}")
            return

    ok, message, config = DocumentIndexService.create_index_config(
        document.id,
        owner.id,
        embedding_model=embedding_model,
        chunk_size=500,
        chunk_overlap=50,
    )
    if not ok or not config:
        print(f"[policy] 索引配置失败：{message}")
        return

    ok, message, task = DocumentIndexService.create_index_task(
        document.id, config.id, owner.id
    )
    if not ok or not task:
        print(f"[policy] 索引任务创建失败：{message}")
        return

    ok, message = DocumentIndexService.run_index_task(task.id)
    if ok:
        print(f"[policy] 索引完成：{message}")
    else:
        print(f"[policy] 索引失败：{message}")
        print("          knowledge_search 将使用本地政策文件降级检索。")


def index_policy(assistant, owner) -> None:
    """Upload and index all demo policy documents."""
    for policy_file in POLICY_FILES:
        print(f"[policy] 处理 {policy_file.name}")
        index_policy_file(assistant, owner, policy_file)


def index_products(assistant) -> None:
    """Index products into the assistant's KB (the KB search_product reads)."""
    from app.models.commerce import Product

    knowledge_base = assistant.knowledge_base
    if not knowledge_base:
        print("[products] 助手未绑定知识库，跳过")
        return
    if Product.query.count() == 0:
        print("[products] 商品目录为空，跳过")
        return

    from scripts.index_products import index_products_into_kb

    try:
        count = index_products_into_kb(knowledge_base, reset=True)
    except Exception as exc:
        print(f"[products] 索引失败（{type(exc).__name__}）：{exc}")
        print("           search_product 将回落到商品目录数据库查询。")
        return
    print(f"[products] 已索引 {count} 条 -> 「{knowledge_base.name}」")


def print_summary(owner, assistant) -> None:
    from app.extensions import db
    from app.models.chat import ChatSession
    from app.models.commerce import Order, Product, ReturnPolicy
    from app.models.document import Document
    from app.services.return_policy import ReturnPolicyEngine

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

    policy_doc = None
    knowledge_base = assistant.knowledge_base
    if knowledge_base:
        policy_doc = Document.query.filter_by(
            knowledge_base_id=knowledge_base.id,
            original_filename=POLICY_FILE.name,
            is_deleted=False,
        ).first()

    print("")
    print("=" * 56)
    print("演示环境已准备")
    print(f"  登录           : {owner.username} / admin123（若未修改）")
    print(f"  订单           : {Order.query.count()} 笔")
    print(f"  可退订单       : {', '.join(returnable) if returnable else '无（请重新运行本脚本）'}")
    print(f"  商品           : {Product.query.count()} 个")
    versions = sorted(
        {
            policy.policy_version
            for policy in ReturnPolicy.query.filter_by(active=True).all()
        }
    )
    print(f"  退货规则版本   : {', '.join(versions) if versions else '未加载'}")
    print(f"  政策文档索引   : {policy_doc.status if policy_doc else '未建立'}")
    print(f"  演示会话       : {ChatSession.query.filter_by(assistant_id=assistant.id).count()} 个")
    print(f"  助手           : {assistant.name} (id={assistant.id})")
    print("")
    print(f"  打开           : http://127.0.0.1:5001/assistants/{assistant.id}/chat")
    print("  启动           : python run.py")
    print("  自检           : python scripts/preflight.py")
    print("=" * 56)


def main() -> int:
    args = parse_args()

    try:
        run_seed(args)
    except subprocess.CalledProcessError as exc:
        print(f"[seed] 失败，退出码 {exc.returncode}")
        return exc.returncode or 1

    from app import create_app
    from app.extensions import db
    from app.models.assistant import Assistant
    from app.models.user import User

    app = create_app()
    with app.app_context():
        owner = User.query.filter_by(username=args.username).first()
        other = User.query.filter_by(username=args.other_username).first()
        assistant = Assistant.query.filter_by(
            name=DEFAULT_ASSISTANT_NAME, is_deleted=False
        ).first()
        if not owner or not assistant:
            print("[bootstrap] 演示用户或助手不存在，请检查 seed 输出")
            return 1

        if not args.no_purge_chat:
            purge_demo_chat(assistant, owner, other or owner)

        if not args.skip_index:
            if not vector_stack_ready():
                print("[index] Milvus/Ollama 未同时就绪，跳过向量索引")
                print("        knowledge_search 使用本地政策降级；商品查询使用数据库目录。")
            else:
                if not args.skip_policy:
                    index_policy(assistant, owner)
                if not args.skip_products:
                    index_products(assistant)
        else:
            print("[index] 已跳过向量索引；knowledge_search 将使用本地政策降级检索")

        print_summary(owner, assistant)
        db.session.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
