"""Seed the small demo commerce dataset.

Usage:
    python scripts/seed_demo.py --username admin
    python scripts/seed_demo.py --username admin --reset

The script is idempotent: existing order numbers / SKUs are left untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app  # noqa: E402
from app.config import Config  # noqa: E402
from app.extensions import db  # noqa: E402
from app.models.agent import PendingAction  # noqa: E402
from app.models.assistant import Assistant, AssistantKnowledgeBaseConfig  # noqa: E402
from app.models.commerce import (  # noqa: E402
    Inventory,
    Order,
    OrderItem,
    ReturnRequest,
    Shipment,
)
from app.models.knowledge_base import KnowledgeBase  # noqa: E402
from app.models.role import Permission, Role  # noqa: E402
from app.models.user import User  # noqa: E402

DATA_FILE = PROJECT_ROOT / "data" / "demo_seed.json"

AFTER_SALES_PERMISSIONS = [
    ("after_sales.read", "after_sales", "read", "查询订单、物流、库存"),
    ("after_sales.write", "after_sales", "write", "发起售后动作"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed demo commerce data")
    parser.add_argument("--username", default="admin", help="user that owns the demo orders")
    parser.add_argument("--other-username", default="demo_other", help="user used for ownership tests")
    parser.add_argument("--reset", action="store_true", help="delete and re-create demo orders")
    return parser.parse_args()


def ensure_permissions() -> None:
    permissions = []
    for name, resource, action, description in AFTER_SALES_PERMISSIONS:
        permission = Permission.query.filter_by(name=name).first()
        if not permission:
            permission = Permission(
                name=name,
                resource=resource,
                action=action,
                description=description,
            )
            db.session.add(permission)
        permissions.append(permission)
    db.session.commit()

    for role_name in ("superadmin", "admin", "user"):
        role = Role.query.filter_by(name=role_name).first()
        if not role:
            continue
        for permission in permissions:
            if permission not in role.permissions:
                role.permissions.append(permission)
    db.session.commit()


def get_or_create_user(username: str, *, password: str, superuser: bool = False) -> User:
    user = User.query.filter_by(username=username).first()
    if user:
        return user

    user = User(
        username=username,
        email=f"{username}@example.com",
        is_superuser=superuser,
    )
    user.set_password(password)
    role_name = "superadmin" if superuser else "user"
    role = Role.query.filter_by(name=role_name).first()
    if role:
        user.roles.append(role)
    db.session.add(user)
    db.session.commit()
    print(f"[users] created {username} / {password}")
    return user


def ensure_demo_assistant(owner: User) -> tuple[KnowledgeBase, Assistant]:
    """Create the demo knowledge base and assistant if they are missing."""
    knowledge_base = KnowledgeBase.query.filter_by(
        name="售后政策", is_deleted=False
    ).first()
    if not knowledge_base:
        knowledge_base = KnowledgeBase(
            name="售后政策",
            description="演示用售后政策知识库，上传 data/policies/refund_policy.md 后索引。",
            default_embedding_model=Config.OLLAMA_EMBEDDING_MODEL,
            created_by=owner.id,
        )
        db.session.add(knowledge_base)
        db.session.commit()

    assistant = Assistant.query.filter_by(name="售后助手", is_deleted=False).first()
    if not assistant:
        assistant = Assistant(
            name="售后助手",
            description="查询订单、物流、库存，并受控发起退货申请。",
            llm_model=Config.LLM_MODEL,
            opening_message="你好，我可以帮你查订单、物流和库存，也可以发起退货申请。",
            created_by=owner.id,
        )
        db.session.add(assistant)
        db.session.flush()

        db.session.add(
            AssistantKnowledgeBaseConfig(
                assistant_id=assistant.id,
                knowledge_base_id=knowledge_base.id,
                retrieval_mode="hybrid",
                top_k=6,
                # The demo works without the local reranker; enable it in the
                # UI once the model path is available.
                enable_rerank=False,
            )
        )
        db.session.commit()

    if not AssistantKnowledgeBaseConfig.query.filter_by(
        assistant_id=assistant.id
    ).first():
        db.session.add(
            AssistantKnowledgeBaseConfig(
                assistant_id=assistant.id,
                knowledge_base_id=knowledge_base.id,
                retrieval_mode="hybrid",
                top_k=6,
                enable_rerank=False,
            )
        )
        db.session.commit()

    return knowledge_base, assistant


def days_ago(value: int | None) -> datetime | None:
    if value is None:
        return None
    return datetime.now() - timedelta(days=int(value))


def upsert_inventory(payload: list[dict]) -> int:
    count = 0
    for item in payload:
        inventory = Inventory.query.filter_by(sku_id=item["sku_id"]).first()
        if not inventory:
            inventory = Inventory(sku_id=item["sku_id"])
            db.session.add(inventory)
            count += 1
        inventory.product_name = item["product_name"]
        inventory.stock = int(item.get("stock", 0))
        inventory.reserved = int(item.get("reserved", 0))
        inventory.warehouse = item.get("warehouse", "WH-1")
    db.session.commit()
    return count


def reset_orders(order_nos: list[str]) -> None:
    orders = Order.query.filter(Order.order_no.in_(order_nos)).all()
    order_ids = [order.id for order in orders]
    if not order_ids:
        return
    PendingAction.query.filter(
        PendingAction.resource_type == "order",
        PendingAction.resource_id.in_(order_ids),
    ).delete(synchronize_session=False)
    ReturnRequest.query.filter(ReturnRequest.order_id.in_(order_ids)).delete(
        synchronize_session=False
    )
    for order in orders:
        db.session.delete(order)
    db.session.commit()
    print(f"[reset] deleted {len(order_ids)} demo orders")


def seed_orders(payload: list[dict], *, owner: User, other: User) -> tuple[int, int]:
    created = 0
    skipped = 0
    for item in payload:
        if Order.query.filter_by(order_no=item["order_no"]).first():
            skipped += 1
            continue

        order_owner = other if item.get("owner") == "other" else owner
        total = round(
            sum(float(line["unit_price"]) * int(line["quantity"]) for line in item["items"]),
            2,
        )
        order = Order(
            order_no=item["order_no"],
            user_id=order_owner.id,
            status=item["status"],
            total_amount=total,
            recipient_name=item.get("recipient_name"),
            recipient_phone=item.get("recipient_phone"),
            address=item.get("address"),
            paid_at=days_ago(item.get("paid_days_ago")),
            delivered_at=days_ago(item.get("delivered_days_ago")),
        )
        db.session.add(order)
        db.session.flush()

        for line in item["items"]:
            db.session.add(
                OrderItem(
                    order_id=order.id,
                    sku_id=line["sku_id"],
                    product_name=line["product_name"],
                    category=line.get("category", "general"),
                    quantity=int(line["quantity"]),
                    unit_price=float(line["unit_price"]),
                    item_status=line.get("item_status", "normal"),
                )
            )
        created += 1

    db.session.commit()
    return created, skipped


def seed_shipments(payload: list[dict]) -> int:
    created = 0
    for item in payload:
        if Shipment.query.filter_by(tracking_no=item["tracking_no"]).first():
            continue
        order = Order.query.filter_by(order_no=item["order_no"]).first()
        if not order:
            continue
        db.session.add(
            Shipment(
                order_id=order.id,
                carrier=item["carrier"],
                tracking_no=item["tracking_no"],
                status=item["status"],
                last_event=item.get("last_event"),
                shipped_at=days_ago(item.get("shipped_days_ago")),
                delivered_at=days_ago(item.get("delivered_days_ago")),
            )
        )
        created += 1
    db.session.commit()
    return created


def main() -> None:
    args = parse_args()
    if not DATA_FILE.exists():
        raise SystemExit(f"seed file not found: {DATA_FILE}")

    with DATA_FILE.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    app = create_app()
    with app.app_context():
        ensure_permissions()
        owner = get_or_create_user(args.username, password="admin123", superuser=(args.username == "admin"))
        other = get_or_create_user(args.other_username, password="demo123")
        knowledge_base, assistant = ensure_demo_assistant(owner)

        if args.reset:
            reset_orders([order["order_no"] for order in payload["orders"]])

        inventories = upsert_inventory(payload["skus"])
        orders, skipped = seed_orders(payload["orders"], owner=owner, other=other)
        shipments = seed_shipments(payload["shipments"])

        print("")
        print("[done] demo data seeded")
        print(f"  owner       : {owner.username} (id={owner.id})")
        print(f"  other user  : {other.username} (id={other.id})")
        print(f"  inventory   : +{inventories}")
        print(f"  orders      : +{orders}, skipped={skipped}")
        print(f"  shipments   : +{shipments}")
        print(f"  knowledge   : {knowledge_base.name} (id={knowledge_base.id})")
        print(f"  assistant   : {assistant.name} (id={assistant.id})")
        print("")
        print(f"Open: /assistants/{assistant.id}/chat")
        print(f"Upload and index: {PROJECT_ROOT / 'data' / 'policies' / 'refund_policy.md'}")
        print("Try these in the assistant chat:")
        print("  1. 订单 202610280001 到哪了？")
        print("  2. 这双鞋还能退吗？")
        print("  3. 帮我退订单 202610280001 的第一件商品，原因是不合适")
        print("  4. 查询订单 202610280006（其他用户的订单，应统一提示未找到）")


if __name__ == "__main__":
    main()
