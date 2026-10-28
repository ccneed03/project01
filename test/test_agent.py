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

from flask import Flask

from app.agents.context import AgentContext
from app.agents.runtime import AgentRuntime
from app.agents.schemas import AgentState, ToolCall
from app.extensions import db
from app.models.agent import AgentRun, PendingAction
from app.models.assistant import Assistant
from app.models.chat import ChatMessage, ChatSession
from app.models.commerce import Order, OrderItem, ReturnRequest
from app.models.user import User
from app.services.agent_service import AgentActionError, AgentService
from app.services.after_sales_service import AfterSalesService
from app.tools.executor import ToolExecutor
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
