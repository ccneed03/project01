"""Unit tests for the agent tool layer and confirmation flow.

Run with:
    python -m unittest discover -s test -v

The tests use an in-memory SQLite database and a fake model gateway, so they
never call the real LLM or Milvus.
"""
from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from flask import Flask

from app.agents.context import AgentContext
from app.agents.runtime import AgentRuntime
from app.agents.schemas import AgentState, ToolCall
from app.extensions import db
from app.models.agent import AgentRun, PendingAction
from app.models.assistant import Assistant, AssistantKnowledgeBaseConfig
from app.models.chat import ChatMessage, ChatSession
from app.models.commerce import Order, OrderItem, Product, ReturnRequest
from app.models.knowledge_base import KnowledgeBase
from app.models.user import User
from app.services.agent_service import AgentActionError, AgentService
from app.services.after_sales_service import AfterSalesService
from app.tools.executor import ToolExecutor
from app.tools.knowledge import KnowledgeSearchArgs, knowledge_search
from app.tools.registry import build_default_registry


class FakeGateway:
    """Scripted gateway: one list of stream events per turn."""

    def __init__(self, turns=None, complete_text="好的，退货申请已提交。"):
        self.turns = list(turns or [])
        self.complete_text = complete_text
        self.complete_calls = 0

    def stream_turn(self, messages, tools=None):
        if not self.turns:
            yield {"type": "content", "content": "完成。"}
            yield {"type": "finish", "reason": "stop"}
            return
        yield from self.turns.pop(0)

    def complete(self, messages, tools=None):
        self.complete_calls += 1
        return {"content": self.complete_text, "tool_calls": [], "finish_reason": "stop"}


class AgentTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = Flask(__name__)
        cls.app.config.update(
            SQLALCHEMY_DATABASE_URI="sqlite://",
            SQLALCHEMY_TRACK_MODIFICATIONS=False,
            TESTING=True,
        )
        db.init_app(cls.app)
        cls.app_context = cls.app.app_context()
        cls.app_context.push()
        db.create_all()

    @classmethod
    def tearDownClass(cls):
        db.session.remove()
        db.drop_all()
        cls.app_context.pop()

    def setUp(self):
        db.session.remove()
        db.drop_all()
        db.create_all()

        self.user = User(username="alice", email="alice@example.com")
        self.user.set_password("secret")
        self.other = User(username="bob", email="bob@example.com")
        self.other.set_password("secret")
        db.session.add_all([self.user, self.other])
        db.session.commit()

        self.assistant = Assistant(
            name="售后助手",
            llm_model="deepseek-flash",
            created_by=self.user.id,
        )
        db.session.add(self.assistant)
        db.session.commit()

        self.chat_session = ChatSession(
            user_id=self.user.id,
            assistant_id=self.assistant.id,
            title="测试会话",
        )
        db.session.add(self.chat_session)
        db.session.commit()

        self.registry = build_default_registry()
        self.executor = ToolExecutor(self.registry)
        self.context = AgentContext(
            user_id=self.user.id,
            session_id=self.chat_session.id,
            trace_id="trace-test",
            run_id="run-test",
            assistant_id=self.assistant.id,
            permissions={"after_sales.read", "after_sales.write", "chat.read"},
        )
        self.state = AgentState(messages=[{"role": "user", "content": "测试"}])

    # ------------------------------------------------------------------ helpers
    def make_order(
        self,
        *,
        user=None,
        order_no="202610280001",
        delivered_days_ago=2,
        status="delivered",
        item_status="normal",
        category="鞋",
        price=299.0,
    ) -> tuple[Order, OrderItem]:
        order = Order(
            order_no=order_no,
            user_id=(user or self.user).id,
            status=status,
            total_amount=price,
            recipient_phone="13800138000",
            paid_at=datetime.now() - timedelta(days=delivered_days_ago + 2),
            delivered_at=(
                datetime.now() - timedelta(days=delivered_days_ago)
                if status == "delivered"
                else None
            ),
        )
        db.session.add(order)
        db.session.flush()
        item = OrderItem(
            order_id=order.id,
            sku_id="SHOE-001",
            product_name="轻量跑鞋",
            category=category,
            quantity=1,
            unit_price=price,
            item_status=item_status,
        )
        db.session.add(item)
        db.session.commit()
        return order, item

    def call(self, name: str, arguments: dict, call_id="call_1") -> ToolCall:
        return ToolCall(
            id=call_id,
            name=name,
            arguments=arguments,
            raw_arguments=json.dumps(arguments, ensure_ascii=False),
        )

    def make_product(
        self,
        *,
        sku_id="SHOE-001",
        product_name="轻量跑鞋 白色 42码",
        category="鞋",
        description="优质商品，材质与规格详见商品详情。",
        return_note="自签收 7 天内支持无理由退货。",
    ) -> Product:
        product = Product(
            sku_id=sku_id,
            product_name=product_name,
            category=category,
            brand="云步",
            description=description,
            details={"材质": "工程网布 + EVA 中底", "尺码": "42"},
            care_info="软毛刷蘸中性洗涤剂轻刷，阴凉晾干，勿机洗。",
            delivery_note="48 小时内发货，默认顺丰。",
            return_note=return_note,
        )
        db.session.add(product)
        db.session.commit()
        return product

    # -------------------------------------------------------------------- tests
    def test_query_order_returns_safe_summary(self):
        order, _ = self.make_order()
        result = self.executor.execute(
            self.call("query_order", {"order_no": order.order_no}),
            self.context,
            self.state,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.code, "OK")
        self.assertIn(order.order_no, result.display_text)
        self.assertEqual(result.data["recipient_phone"], "138****8000")
        self.assertNotIn("address", result.data)

    def test_query_order_rejects_other_users_order(self):
        order, _ = self.make_order(user=self.other, order_no="202610280006")
        result = self.executor.execute(
            self.call("query_order", {"order_no": order.order_no}),
            self.context,
            self.state,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "NOT_FOUND")
        self.assertEqual(result.display_text, "未找到该订单，请确认订单号")

    def test_query_order_requires_order_no_when_no_active_order(self):
        result = self.executor.execute(
            self.call("query_order", {}),
            self.context,
            self.state,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "NEED_ORDER_NO")

    def test_search_product_by_sku(self):
        self.make_product()
        result = self.executor.execute(
            self.call("search_product", {"sku_id": "shoe-001"}),
            self.context,
            self.state,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.code, "OK")
        self.assertEqual(len(result.data["products"]), 1)
        self.assertEqual(result.data["products"][0]["sku_id"], "SHOE-001")
        self.assertIn("SHOE-001", result.display_text)
        self.assertEqual(
            result.data["products"][0]["details"]["材质"], "工程网布 + EVA 中底"
        )

    def test_search_product_by_keyword(self):
        self.make_product(sku_id="SHOE-001", product_name="轻量跑鞋 白色 42码")
        self.make_product(sku_id="SHOE-002", product_name="轻量跑鞋 黑色 41码")
        self.make_product(sku_id="BAG-001", product_name="通勤双肩包 深蓝色", category="包")
        result = self.executor.execute(
            self.call("search_product", {"query": "跑鞋"}),
            self.context,
            self.state,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.code, "OK")
        self.assertEqual(len(result.data["products"]), 2)

    def test_search_product_by_category(self):
        self.make_product(sku_id="SHOE-001", category="鞋")
        self.make_product(sku_id="SHOE-002", category="鞋")
        self.make_product(sku_id="BAG-001", category="包")
        result = self.executor.execute(
            self.call("search_product", {"category": "鞋"}),
            self.context,
            self.state,
        )
        self.assertTrue(result.ok)
        self.assertEqual(len(result.data["products"]), 2)

    def test_search_product_description_fallback(self):
        """商品没被订单/库存覆盖时，也可通过目录说明检索到。"""
        self.make_product(sku_id="JACKET-001", product_name="轻薄防风外套 黑色 L码", category="服装")
        result = self.executor.execute(
            self.call("search_product", {"query": "外套"}),
            self.context,
            self.state,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.data["products"][0]["sku_id"], "JACKET-001")

    def test_search_product_not_found(self):
        result = self.executor.execute(
            self.call("search_product", {"sku_id": "NOPE-999"}),
            self.context,
            self.state,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "NOT_FOUND")

    def test_search_product_no_keyword(self):
        result = self.executor.execute(
            self.call("search_product", {}),
            self.context,
            self.state,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "NEED_QUERY")

    def test_product_filter_expr_builds_scalar_filter(self):
        from app.tools.product import _product_filter_expr, _quote_attr

        self.assertEqual(
            _product_filter_expr(None, None), 'doc_type == "product"'
        )
        self.assertEqual(
            _product_filter_expr("鞋", "SHOE-001"),
            'doc_type == "product" and category == "鞋" and sku_id == "SHOE-001"',
        )
        # quotes in attribute values are escaped, not injected
        self.assertEqual(
            _quote_attr('a"b'), '"a\\"b"'
        )

    def test_search_product_falls_back_to_db_when_retrieval_unavailable(self):
        """语义检索不可用时，应安静退回权威数据库目录，不影响闭环。"""
        from unittest import mock

        from app.models.knowledge_base import KnowledgeBase

        self.make_product(sku_id="SHOE-001", category="鞋")
        kb = KnowledgeBase(name="产品库", description="语义检索", created_by=self.user.id)
        db.session.add(kb)
        db.session.commit()

        context = AgentContext(
            user_id=self.user.id,
            session_id=self.chat_session.id,
            trace_id="trace-test",
            run_id="run-test",
            assistant_id=self.assistant.id,
            knowledge_base_id=kb.id,
            permissions={"after_sales.read"},
        )
        with mock.patch(
            "app.services.retrieval_service.RetrievalService.search",
            side_effect=RuntimeError("Milvus 不可用"),
        ):
            result = self.executor.execute(
                self.call("search_product", {"query": "跑鞋"}),
                context,
                self.state,
            )
        self.assertTrue(result.ok)
        self.assertEqual(result.code, "OK")
        self.assertEqual(result.data["products"][0]["sku_id"], "SHOE-001")

    def test_search_product_uses_semantic_hits_when_retrieval_available(self):
        """向量层命中但内容不可信：只回落到 *权威目录* 按 sku_id 对齐。"""
        from unittest import mock

        from app.models.knowledge_base import KnowledgeBase

        self.make_product(sku_id="SHOE-001", category="鞋")
        kb = KnowledgeBase(
            name="产品库", description="语义检索", created_by=self.user.id
        )
        db.session.add(kb)
        db.session.commit()

        context = AgentContext(
            user_id=self.user.id,
            session_id=self.chat_session.id,
            trace_id="trace-test",
            run_id="run-test",
            assistant_id=self.assistant.id,
            knowledge_base_id=kb.id,
            permissions={"after_sales.read"},
        )
        hits = [
            {
                "text": "轻量缓震跑鞋",
                "sku_id": "SHOE-001",
                "score": 0.9,
            }
        ]
        with mock.patch(
            "app.services.retrieval_service.RetrievalService.search",
            return_value=hits,
        ):
            result = self.executor.execute(
                self.call("search_product", {"query": "雨后路滑穿什么鞋"}),
                context,
                self.state,
            )
        self.assertTrue(result.ok)
        self.assertEqual(result.code, "OK")
        self.assertEqual(result.data["products"][0]["sku_id"], "SHOE-001")

    def test_product_index_defaults_to_assistant_kb(self):
        from scripts.index_products import resolve_knowledge_base

        knowledge_base = KnowledgeBase(
            name="售后政策",
            created_by=self.user.id,
            default_embedding_model="qwen3-embedding:0.6b",
        )
        db.session.add(knowledge_base)
        db.session.commit()
        db.session.add(
            AssistantKnowledgeBaseConfig(
                assistant_id=self.assistant.id,
                knowledge_base_id=knowledge_base.id,
            )
        )
        db.session.commit()

        resolved, source = resolve_knowledge_base()
        self.assertEqual(resolved.id, knowledge_base.id)
        self.assertIn(self.assistant.name, source)

    def test_knowledge_search_falls_back_to_local_policy(self):
        knowledge_base = KnowledgeBase(
            name="售后政策",
            created_by=self.user.id,
            default_embedding_model="qwen3-embedding:0.6b",
        )
        db.session.add(knowledge_base)
        db.session.commit()
        context = AgentContext(
            user_id=self.user.id,
            session_id=self.chat_session.id,
            trace_id="trace-test",
            run_id="run-test",
            assistant_id=self.assistant.id,
            knowledge_base_id=knowledge_base.id,
            permissions={"after_sales.read"},
        )

        with patch(
            "app.services.retrieval_service.RetrievalService.search",
            side_effect=RuntimeError("milvus down"),
        ):
            result = knowledge_search(
                KnowledgeSearchArgs(query="这双鞋还能退吗？"),
                context,
                self.state,
                self.call("knowledge_search", {"query": "这双鞋还能退吗？"}),
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.code, "OK_DEGRADED")
        self.assertTrue(result.data["degraded"])
        self.assertTrue(result.sources)
        document = result.sources[0]["document"]
        self.assertEqual(document["original_filename"], "refund_policy.md")
        self.assertEqual(document["source"], "local_fallback")
        self.assertTrue(
            any("7" in snippet["text"] or "七" in snippet["text"] for snippet in result.data["snippets"])
        )

    def test_knowledge_search_prefers_vector_results(self):
        knowledge_base = KnowledgeBase(
            name="售后政策",
            created_by=self.user.id,
            default_embedding_model="qwen3-embedding:0.6b",
        )
        db.session.add(knowledge_base)
        db.session.commit()
        context = AgentContext(
            user_id=self.user.id,
            session_id=self.chat_session.id,
            trace_id="trace-test",
            run_id="run-test",
            assistant_id=self.assistant.id,
            knowledge_base_id=knowledge_base.id,
            permissions={"after_sales.read"},
        )
        hits = [
            {
                "text": "向量检索命中的政策片段：签收后 7 天内可申请无理由退货。",
                "document": {"original_filename": "refund_policy.pdf"},
                "chunk_index": 3,
                "score": 0.91,
            }
        ]

        with patch(
            "app.services.retrieval_service.RetrievalService.search",
            return_value=hits,
        ):
            result = knowledge_search(
                KnowledgeSearchArgs(query="还能退吗？"),
                context,
                self.state,
                self.call("knowledge_search", {"query": "还能退吗？"}),
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.code, "OK")
        self.assertNotIn("degraded", result.data)
        self.assertEqual(result.sources[0]["document"]["original_filename"], "refund_policy.pdf")

    def test_permission_denied_is_fuzzy(self):
        order, _ = self.make_order()
        denied_context = AgentContext(
            user_id=self.user.id,
            session_id=self.chat_session.id,
            trace_id="trace-test",
            permissions=set(),
        )
        result = self.executor.execute(
            self.call("query_order", {"order_no": order.order_no}),
            denied_context,
            self.state,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "DENIED")

    def test_invalid_arguments_are_rejected(self):
        result = self.executor.execute(
            self.call("prepare_return", {"order_no": "202610280001", "item_ids": []}),
            self.context,
            self.state,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "INVALID_ARGUMENTS")

    def test_prepare_return_creates_pending_without_mutation(self):
        order, item = self.make_order()
        result = self.executor.execute(
            self.call(
                "prepare_return",
                {
                    "order_no": order.order_no,
                    "item_ids": [item.id],
                    "reason": "不合适",
                },
            ),
            self.context,
            self.state,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.code, "CONFIRMATION_REQUIRED")
        self.assertEqual(PendingAction.query.count(), 1)
        self.assertEqual(ReturnRequest.query.count(), 0)
        self.assertEqual(db.session.get(OrderItem, item.id).item_status, "normal")

    def test_prepare_return_rejects_expired_window(self):
        order, item = self.make_order(delivered_days_ago=10)
        result = self.executor.execute(
            self.call(
                "prepare_return",
                {"order_no": order.order_no, "item_ids": [item.id]},
            ),
            self.context,
            self.state,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "NOT_ELIGIBLE")
        self.assertEqual(PendingAction.query.count(), 0)

    def test_prepare_return_rejects_custom_category(self):
        order, item = self.make_order(category="定制")
        result = self.executor.execute(
            self.call(
                "prepare_return",
                {"order_no": order.order_no, "item_ids": [item.id]},
            ),
            self.context,
            self.state,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "NOT_ELIGIBLE")
        self.assertEqual(PendingAction.query.count(), 0)

    def test_confirm_executes_once(self):
        order, item = self.make_order()
        result = self.executor.execute(
            self.call(
                "prepare_return",
                {"order_no": order.order_no, "item_ids": [item.id]},
            ),
            self.context,
            self.state,
        )
        pending_id = result.pending_action_id
        state = AgentState(
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "我要退货"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "prepare_return", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": '{"code": "CONFIRMATION_REQUIRED"}',
                },
            ],
            step_count=1,
        )
        db.session.add(
            AgentRun(
                run_id="run-test",
                trace_id="trace-test",
                user_id=self.user.id,
                session_id=self.chat_session.id,
                assistant_id=self.assistant.id,
                status="awaiting_confirmation",
                step_count=1,
                input="我要退货",
                state_json=state.to_dict(),
            )
        )
        db.session.commit()
        gateway = FakeGateway()

        first = AgentService.confirm_action(pending_id, self.user.id, gateway=gateway)
        self.assertTrue(first["success"])
        self.assertEqual(ReturnRequest.query.count(), 1)
        self.assertEqual(gateway.complete_calls, 1)

        run = AgentRun.query.filter_by(run_id="run-test").first()
        tool_messages = [
            message
            for message in run.state_json["messages"]
            if message.get("role") == "tool" and message.get("tool_call_id") == "call_1"
        ]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("OK", tool_messages[0]["content"])

        with self.assertRaises(AgentActionError) as raised:
            AgentService.confirm_action(pending_id, self.user.id, gateway=gateway)
        self.assertEqual(raised.exception.code, "ALREADY_PROCESSED")
        self.assertEqual(ReturnRequest.query.count(), 1)

    def test_cancel_does_not_execute(self):
        order, item = self.make_order()
        result = self.executor.execute(
            self.call(
                "prepare_return",
                {"order_no": order.order_no, "item_ids": [item.id]},
            ),
            self.context,
            self.state,
        )
        cancelled = AgentService.cancel_action(result.pending_action_id, self.user.id)
        self.assertTrue(cancelled["success"])
        self.assertEqual(ReturnRequest.query.count(), 0)
        self.assertEqual(
            db.session.get(PendingAction, result.pending_action_id).status,
            "cancelled",
        )

    def test_prepare_after_cancel_creates_new_pending(self):
        order, item = self.make_order()
        first = self.executor.execute(
            self.call(
                "prepare_return",
                {"order_no": order.order_no, "item_ids": [item.id], "reason": "不合适"},
            ),
            self.context,
            self.state,
        )
        self.assertTrue(first.ok)
        AgentService.cancel_action(first.pending_action_id, self.user.id)

        second = self.executor.execute(
            self.call(
                "prepare_return",
                {"order_no": order.order_no, "item_ids": [item.id], "reason": "不合适"},
            ),
            self.context,
            self.state,
        )
        self.assertTrue(second.ok)
        self.assertNotEqual(first.pending_action_id, second.pending_action_id)
        self.assertEqual(
            db.session.get(PendingAction, second.pending_action_id).status,
            "pending",
        )

    def test_prepare_after_expiry_creates_new_pending(self):
        order, item = self.make_order()
        first = self.executor.execute(
            self.call(
                "prepare_return",
                {"order_no": order.order_no, "item_ids": [item.id], "reason": "不合适"},
            ),
            self.context,
            self.state,
        )
        self.assertTrue(first.ok)
        pending = db.session.get(PendingAction, first.pending_action_id)
        pending.expires_at = datetime.now() - timedelta(minutes=1)
        db.session.commit()

        AgentService.pending_actions(self.chat_session.id, self.user.id)
        self.assertEqual(
            db.session.get(PendingAction, first.pending_action_id).status,
            "expired",
        )

        second = self.executor.execute(
            self.call(
                "prepare_return",
                {"order_no": order.order_no, "item_ids": [item.id], "reason": "不合适"},
            ),
            self.context,
            self.state,
        )
        self.assertTrue(second.ok)
        self.assertNotEqual(first.pending_action_id, second.pending_action_id)

    def test_prepare_while_pending_is_idempotent(self):
        order, item = self.make_order()
        arguments = {
            "order_no": order.order_no,
            "item_ids": [item.id],
            "reason": "不合适",
        }
        first = self.executor.execute(
            self.call("prepare_return", arguments), self.context, self.state
        )
        second = self.executor.execute(
            self.call("prepare_return", arguments), self.context, self.state
        )
        self.assertTrue(first.ok)
        self.assertTrue(second.ok)
        self.assertEqual(first.pending_action_id, second.pending_action_id)

    def test_other_user_cannot_confirm_pending_action(self):
        order, item = self.make_order()
        result = self.executor.execute(
            self.call(
                "prepare_return",
                {"order_no": order.order_no, "item_ids": [item.id]},
            ),
            self.context,
            self.state,
        )
        with self.assertRaises(AgentActionError) as raised:
            AgentService.confirm_action(
                result.pending_action_id, self.other.id, gateway=FakeGateway()
            )
        self.assertEqual(raised.exception.code, "NOT_FOUND")
        self.assertEqual(ReturnRequest.query.count(), 0)

    def test_commit_return_is_not_exposed_to_model(self):
        names = {
            schema["function"]["name"]
            for schema in self.registry.planning_schemas()
        }
        self.assertNotIn("commit_return", names)
        self.assertIsNone(self.registry.get("commit_return"))

    def test_expired_action_is_rejected(self):
        order, item = self.make_order()
        pending = AfterSalesService.prepare_return(
            order=order,
            item_ids=[item.id],
            reason="",
            user_id=self.user.id,
            session_id=self.chat_session.id,
            run_id="run-test",
            tool_call_id="call-expired",
        )
        pending.expires_at = datetime.now() - timedelta(minutes=1)
        db.session.commit()

        with self.assertRaises(AgentActionError) as raised:
            AgentService.confirm_action(pending.id, self.user.id, gateway=FakeGateway())
        self.assertEqual(raised.exception.code, "EXPIRED")
        self.assertEqual(ReturnRequest.query.count(), 0)

    def test_runtime_executes_tool_then_answers(self):
        order, _ = self.make_order()
        turns = [
            [
                {"type": "content", "content": "我查一下。"},
                {
                    "type": "tool_call_delta",
                    "index": 0,
                    "id": "call_1",
                    "name": "query_order",
                    "arguments": '{"order_no": "',
                },
                {
                    "type": "tool_call_delta",
                    "index": 0,
                    "arguments": f'{order.order_no}"}}',
                },
                {"type": "finish", "reason": "tool_calls"},
            ],
            [
                {"type": "content", "content": "订单已签收。"},
                {"type": "finish", "reason": "stop"},
            ],
        ]
        gateway = FakeGateway(turns)
        state = AgentState(
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "订单到哪了"},
            ]
        )
        runtime = AgentRuntime(gateway, self.executor, self.registry, max_steps=5)
        events = list(runtime.run(state, self.context))
        types = [event["type"] for event in events]

        self.assertIn("tool_start", types)
        self.assertIn("tool_result", types)
        self.assertEqual(events[-1]["type"], "end")
        self.assertEqual(events[-1]["content"], "订单已签收。")
        tool_start = next(event for event in events if event["type"] == "tool_start")
        self.assertTrue(tool_start["discard_content"])

        roles = [message["role"] for message in state.messages]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant"])
        assistant_message = state.messages[2]
        self.assertEqual(assistant_message["tool_calls"][0]["function"]["name"], "query_order")

    def test_runtime_pauses_at_confirmation_with_placeholder(self):
        order, item = self.make_order()
        arguments = json.dumps(
            {"order_no": order.order_no, "item_ids": [item.id], "reason": "不合适"},
            ensure_ascii=False,
        )
        turns = [
            [
                {
                    "type": "tool_call_delta",
                    "index": 0,
                    "id": "call_pending",
                    "name": "prepare_return",
                    "arguments": arguments,
                },
                {"type": "finish", "reason": "tool_calls"},
            ]
        ]
        gateway = FakeGateway(turns)
        state = AgentState(
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "我要退货"},
            ]
        )
        runtime = AgentRuntime(gateway, self.executor, self.registry, max_steps=5)
        events = list(runtime.run(state, self.context))

        self.assertEqual(events[-1]["type"], "confirmation_required")
        self.assertIsNotNone(state.pending_action_id)
        self.assertEqual(
            [message["role"] for message in state.messages],
            ["system", "user", "assistant", "tool"],
        )
        placeholder = state.messages[-1]
        self.assertEqual(placeholder["tool_call_id"], "call_pending")
        self.assertIn("CONFIRMATION_REQUIRED", placeholder["content"])

    def test_runtime_stops_at_max_steps(self):
        class LoopGateway:
            def __init__(self):
                self.index = 0

            def stream_turn(self, messages, tools=None):
                self.index += 1
                yield {
                    "type": "tool_call_delta",
                    "index": 0,
                    "id": f"call_{self.index}",
                    "name": "check_inventory",
                    "arguments": '{"sku_id": "SHOE-001"}',
                }
                yield {"type": "finish", "reason": "tool_calls"}

            def complete(self, messages, tools=None):
                return {"content": "", "tool_calls": [], "finish_reason": "stop"}

        state = AgentState(messages=[{"role": "user", "content": "loop"}])
        runtime = AgentRuntime(LoopGateway(), self.executor, self.registry, max_steps=3)
        events = list(runtime.run(state, self.context))
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(state.step_count, 3)

    def test_demo_seed_is_idempotent(self):
        import json

        from scripts.seed_demo import (
            DATA_FILE,
            seed_orders,
            seed_shipments,
            upsert_inventory,
        )

        payload = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        self.assertEqual(upsert_inventory(payload["skus"]), 8)

        created, skipped = seed_orders(
            payload["orders"],
            owner=self.user,
            other=self.other,
        )
        self.assertEqual(created, 8)
        self.assertEqual(skipped, 0)
        self.assertEqual(seed_shipments(payload["shipments"]), 7)

        own_order = Order.query.filter_by(order_no="202610280001").first()
        other_order = Order.query.filter_by(order_no="202610280006").first()
        self.assertEqual(own_order.user_id, self.user.id)
        self.assertEqual(other_order.user_id, self.other.id)

        created_again, skipped_again = seed_orders(
            payload["orders"],
            owner=self.user,
            other=self.other,
        )
        self.assertEqual(created_again, 0)
        self.assertEqual(skipped_again, 8)

    def test_save_state_by_run_id(self):
        run = AgentRun(
            run_id="run-save",
            trace_id="trace-save",
            user_id=self.user.id,
            session_id=self.chat_session.id,
            assistant_id=self.assistant.id,
            status="running",
            step_count=0,
            input="test",
            state_json={},
        )
        db.session.add(run)
        db.session.commit()

        state = AgentState(
            messages=[{"role": "user", "content": "hi"}],
            step_count=2,
            active_order_id=1,
        )
        AgentService.save_state_by_run_id(
            "run-save", state, status="awaiting_confirmation"
        )
        db.session.expire_all()

        loaded = AgentRun.query.filter_by(run_id="run-save").first()
        self.assertEqual(loaded.status, "awaiting_confirmation")
        self.assertEqual(loaded.step_count, 2)
        self.assertEqual(loaded.state_json["active_order_id"], 1)
        self.assertEqual(loaded.state_json["messages"][0]["content"], "hi")

    def test_start_run_persists_history_and_state(self):
        db.session.add(
            ChatMessage(session_id=self.chat_session.id, role="user", content="上一轮问题")
        )
        db.session.add(
            ChatMessage(session_id=self.chat_session.id, role="assistant", content="上一轮回答")
        )
        db.session.commit()

        run, context, state = AgentService.start_run(
            user=self.user,
            session=self.chat_session,
            assistant=self.assistant,
            query="这双鞋能退吗？",
        )
        self.assertEqual(run.status, "running")
        self.assertEqual(context.knowledge_base_id, None)
        self.assertEqual(state.messages[0]["role"], "system")
        self.assertEqual(state.messages[-1]["content"], "这双鞋能退吗？")
        self.assertIn("上一轮问题", [message["content"] for message in state.messages])
        self.assertEqual(ChatMessage.query.filter_by(session_id=self.chat_session.id).count(), 3)

    def test_start_run_restores_active_order_from_last_run(self):
        order, _ = self.make_order()
        db.session.add(
            AgentRun(
                run_id="previous-run",
                trace_id="previous-trace",
                user_id=self.user.id,
                session_id=self.chat_session.id,
                assistant_id=self.assistant.id,
                status="completed",
                step_count=0,
                input="上一轮",
                state_json={
                    "messages": [],
                    "active_order_id": order.id,
                    "candidate_orders": [order.id],
                },
            )
        )
        db.session.commit()

        _, _, state = AgentService.start_run(
            user=self.user,
            session=self.chat_session,
            assistant=self.assistant,
            query="这个订单还能退吗？",
        )
        self.assertEqual(state.active_order_id, order.id)
        self.assertEqual(state.candidate_orders, [order.id])


if __name__ == "__main__":
    unittest.main()