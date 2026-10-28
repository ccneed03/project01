"""Business orchestration between HTTP routes, the runtime and the database."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from app.agents.context import AgentContext
from app.agents.prompts import build_system_prompt
from app.agents.schemas import AgentState
from app.config import Config
from app.extensions import db
from app.models.agent import AgentRun, AgentStep, PendingAction
from app.models.assistant import Assistant
from app.models.chat import ChatMessage, ChatSession
from app.services.after_sales_service import AfterSalesError, AfterSalesService


class AgentActionError(Exception):
    """A confirm/cancel failure with an HTTP-friendly status."""

    def __init__(self, code: str, message: str, http_status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


class AgentService:
    """Create runs, persist state and execute confirmations."""

    @staticmethod
    def permissions_for(user) -> set[str]:
        if getattr(user, "is_superuser", False):
            return {"*"}
        permissions: set[str] = set()
        for role in user.roles:
            for permission in role.permissions:
                permissions.add(f"{permission.resource}.{permission.action}")
        return permissions

    @staticmethod
    def build_context(
        *,
        user,
        session: ChatSession,
        assistant,
        run_id: str,
        trace_id: str,
    ) -> AgentContext:
        binding = assistant.primary_knowledge_base_config if assistant else None
        knowledge_base = binding.knowledge_base if binding else None
        retrieval_config = {}
        if binding:
            retrieval_config = {
                "retrieval_mode": binding.retrieval_mode,
                "top_k": binding.top_k,
                "enable_rerank": binding.enable_rerank,
                "rerank_model": binding.rerank_model,
                "score_threshold": binding.score_threshold,
            }
        return AgentContext(
            user_id=user.id,
            session_id=session.id,
            trace_id=trace_id,
            run_id=run_id,
            assistant_id=assistant.id if assistant else None,
            knowledge_base_id=knowledge_base.id if knowledge_base else None,
            permissions=AgentService.permissions_for(user),
            retrieval_config=retrieval_config,
        )

    @staticmethod
    def start_run(
        *,
        user,
        session: ChatSession,
        assistant,
        query: str,
    ) -> tuple[AgentRun, AgentContext, AgentState]:
        """Persist the user message, create an AgentRun and build messages."""
        history_rows = (
            ChatMessage.query.filter_by(session_id=session.id)
            .order_by(ChatMessage.created_at.desc())
            .limit(Config.AGENT_HISTORY_LIMIT)
            .all()
        )
        history = [
            {"role": message.role, "content": message.content}
            for message in reversed(history_rows)
            if message.role in {"user", "assistant"} and message.content
        ]

        # Restore the order context from the previous run so follow-up
        # questions like “这个订单能退吗” can use the same order.
        last_run = (
            AgentRun.query.filter_by(session_id=session.id)
            .order_by(AgentRun.id.desc())
            .first()
        )
        last_state = AgentState.from_dict(last_run.state_json) if last_run else None

        user_message = ChatMessage(session_id=session.id, role="user", content=query)
        db.session.add(user_message)

        run_id = uuid.uuid4().hex
        trace_id = uuid.uuid4().hex
        run = AgentRun(
            run_id=run_id,
            trace_id=trace_id,
            user_id=user.id,
            session_id=session.id,
            assistant_id=assistant.id if assistant else None,
            status="running",
            step_count=0,
            input=query,
            state_json={},
        )
        db.session.add(run)

        state = AgentState(
            messages=[
                {"role": "system", "content": build_system_prompt(assistant)},
                *history,
                {"role": "user", "content": query},
            ],
            active_order_id=last_state.active_order_id if last_state else None,
            candidate_orders=last_state.candidate_orders if last_state else [],
        )
        run.state_json = state.to_dict()
        session.updated_at = datetime.now()
        db.session.commit()

        context = AgentService.build_context(
            user=user,
            session=session,
            assistant=assistant,
            run_id=run_id,
            trace_id=trace_id,
        )
        return run, context, state

    @staticmethod
    def save_state(run: AgentRun, state: AgentState, status: str | None = None) -> None:
        run.state_json = state.to_dict()
        run.step_count = state.step_count
        if status:
            run.status = status
        db.session.commit()

    @staticmethod
    def save_state_by_run_id(
        run_id: str,
        state: AgentState,
        status: str | None = None,
    ) -> AgentRun | None:
        """Persist state from inside an SSE generator.

        ORM instances created in the view function are detached once Flask
        tears down the request context; always re-load the run in the
        generator's own session before writing.
        """
        run = AgentRun.query.filter_by(run_id=run_id).first()
        if not run:
            return None
        run.state_json = state.to_dict()
        run.step_count = state.step_count
        if status:
            run.status = status
        db.session.commit()
        return run

    @staticmethod
    def record_step(
        *,
        run_id: str,
        step_index: int,
        tool_name: str,
        tool_call_id: str | None,
        arguments: dict[str, Any],
        result_code: str,
        ok: bool,
        duration_ms: int,
    ) -> None:
        db.session.add(
            AgentStep(
                run_id=run_id,
                step_index=step_index,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                args_json=arguments or {},
                result_code=result_code,
                ok=ok,
                duration_ms=duration_ms,
            )
        )
        db.session.commit()

    @staticmethod
    def save_assistant_message(
        *,
        session_id: int,
        content: str,
        references: list[dict[str, Any]] | None = None,
    ) -> ChatMessage:
        message = ChatMessage(
            session_id=session_id,
            role="assistant",
            content=content,
            references=references,
        )
        db.session.add(message)
        session = db.session.get(ChatSession, session_id)
        if session:
            session.updated_at = datetime.now()
        db.session.commit()
        return message

    @staticmethod
    def mark_run_failed(run: AgentRun, message: str) -> None:
        run.status = "failed"
        db.session.commit()

    @staticmethod
    def mark_run_failed_by_id(run_id: str, message: str) -> None:
        run = AgentRun.query.filter_by(run_id=run_id).first()
        if run:
            run.status = "failed"
            db.session.commit()

    @staticmethod
    def confirm_action(action_id: int, user_id: int, gateway=None) -> dict[str, Any]:
        """Execute a PendingAction exactly once and produce the final answer."""
        pending = db.session.get(PendingAction, action_id)
        if not pending or pending.user_id != user_id:
            raise AgentActionError("NOT_FOUND", "确认动作不存在", 404)

        now = datetime.now()
        if pending.status != "pending":
            raise AgentActionError("ALREADY_PROCESSED", "该动作已处理或已取消", 409)
        if pending.expires_at and pending.expires_at < now:
            pending.status = "expired"
            db.session.commit()
            raise AgentActionError("EXPIRED", "确认已过期，请重新发起", 410)

        # Atomic transition prevents double clicks / parallel requests from
        # executing the same refund twice.
        updated = PendingAction.query.filter_by(
            id=action_id,
            user_id=user_id,
            status="pending",
        ).update(
            {
                "status": "executing",
                "confirmed_at": now,
            },
            synchronize_session=False,
        )
        db.session.commit()
        if updated != 1:
            raise AgentActionError("ALREADY_PROCESSED", "该动作正在处理或已完成", 409)
        db.session.refresh(pending)

        try:
            result = AfterSalesService.commit_return(pending)
        except AfterSalesError as exc:
            pending.status = "failed"
            pending.result_json = {"code": exc.code, "message": exc.message}
            db.session.commit()
            raise AgentActionError(exc.code, exc.message, 409) from exc
        except Exception as exc:
            db.session.rollback()
            pending = db.session.get(PendingAction, action_id)
            if pending:
                pending.status = "failed"
                pending.result_json = {"code": "EXECUTION_FAILED", "message": str(exc)[:300]}
                db.session.commit()
            raise AgentActionError("EXECUTION_FAILED", "退货执行失败，请稍后重试", 500) from exc

        pending.status = "success"
        pending.executed_at = datetime.now()
        pending.result_json = result
        db.session.commit()

        run = AgentRun.query.filter_by(run_id=pending.run_id).first()
        response_text = result["display_text"]
        if run:
            state = AgentState.from_dict(run.state_json)
            AgentService._set_tool_message(
                state,
                pending.tool_call_id,
                json.dumps(
                    {
                        "ok": True,
                        "code": "OK",
                        "display_text": result["display_text"],
                        "data": result,
                    },
                    ensure_ascii=False,
                ),
            )
            state.messages.append(
                {
                    "role": "system",
                    "content": (
                        "系统事件：用户已通过确认卡片确认，退货申请已成功提交。"
                        "请直接告知用户退货单号和退款金额，不要再次要求确认，"
                        "也不要重复“等待确认”之类的前置提示。"
                    ),
                }
            )
            state.step_count += 1
            state.pending_action_id = None

            try:
                if gateway is None:
                    from app.agents.gateway import ChatGateway

                    assistant = (
                        db.session.get(Assistant, run.assistant_id)
                        if run.assistant_id
                        else None
                    )
                    gateway = ChatGateway(model=getattr(assistant, "llm_model", None) or Config.LLM_MODEL)
                completion = gateway.complete(state.messages, tools=None)
                response_text = (completion.get("content") or "").strip() or result["display_text"]
            except Exception:
                # The refund already committed; a model failure must not hide
                # that fact.  Fall back to the deterministic display text.
                response_text = result["display_text"]

            state.messages.append({"role": "assistant", "content": response_text})
            run.state_json = state.to_dict()
            run.step_count = state.step_count
            run.status = "completed"
            db.session.commit()

            AgentService.record_step(
                run_id=run.run_id,
                step_index=state.step_count,
                tool_name="commit_return",
                tool_call_id=pending.tool_call_id,
                arguments={"action_id": pending.id, "order_id": pending.resource_id},
                result_code="OK",
                ok=True,
                duration_ms=0,
            )
            AgentService.save_assistant_message(
                session_id=pending.session_id,
                content=response_text,
            )
        else:
            AgentService.save_assistant_message(
                session_id=pending.session_id,
                content=response_text,
            )

        return {
            "success": True,
            "status": "success",
            "action_id": pending.id,
            "display_text": result["display_text"],
            "response": response_text,
            "run_id": pending.run_id,
            "session_id": pending.session_id,
        }

    @staticmethod
    def cancel_action(action_id: int, user_id: int) -> dict[str, Any]:
        pending = db.session.get(PendingAction, action_id)
        if not pending or pending.user_id != user_id:
            raise AgentActionError("NOT_FOUND", "确认动作不存在", 404)

        updated = PendingAction.query.filter_by(
            id=action_id,
            user_id=user_id,
            status="pending",
        ).update({"status": "cancelled"}, synchronize_session=False)
        db.session.commit()
        if updated != 1:
            raise AgentActionError("ALREADY_PROCESSED", "该动作已处理或已取消", 409)
        db.session.refresh(pending)

        response_text = "已取消本次退货申请，没有产生任何变更。"
        run = AgentRun.query.filter_by(run_id=pending.run_id).first()
        if run:
            state = AgentState.from_dict(run.state_json)
            AgentService._set_tool_message(
                state,
                pending.tool_call_id,
                json.dumps(
                    {"ok": False, "code": "USER_CANCELLED", "display_text": response_text},
                    ensure_ascii=False,
                ),
            )
            state.pending_action_id = None
            run.state_json = state.to_dict()
            run.status = "cancelled"
            db.session.commit()
            AgentService.record_step(
                run_id=run.run_id,
                step_index=state.step_count,
                tool_name="cancel_return",
                tool_call_id=pending.tool_call_id,
                arguments={"action_id": pending.id},
                result_code="USER_CANCELLED",
                ok=True,
                duration_ms=0,
            )

        AgentService.save_assistant_message(session_id=pending.session_id, content=response_text)
        return {
            "success": True,
            "status": "cancelled",
            "action_id": pending.id,
            "display_text": response_text,
            "response": response_text,
            "run_id": pending.run_id,
            "session_id": pending.session_id,
        }

    @staticmethod
    def pending_actions(session_id: int, user_id: int) -> list[dict[str, Any]]:
        now = datetime.now()
        PendingAction.query.filter(
            PendingAction.session_id == session_id,
            PendingAction.user_id == user_id,
            PendingAction.status == "pending",
            PendingAction.expires_at < now,
        ).update({"status": "expired"}, synchronize_session=False)
        db.session.commit()

        actions = (
            PendingAction.query.filter_by(
                session_id=session_id,
                user_id=user_id,
                status="pending",
            )
            .order_by(PendingAction.id.desc())
            .all()
        )
        return [AgentService.serialize_pending(action) for action in actions]

    @staticmethod
    def _set_tool_message(
        state: AgentState,
        tool_call_id: str | None,
        content: str,
    ) -> None:
        """Replace the placeholder from the pause, or append when absent."""
        if tool_call_id:
            for message in state.messages:
                if (
                    message.get("role") == "tool"
                    and message.get("tool_call_id") == tool_call_id
                ):
                    message["content"] = content
                    return
        state.messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        )

    @staticmethod
    def serialize_pending(pending: PendingAction) -> dict[str, Any]:
        display = dict(pending.display_json or {})
        display["action_id"] = pending.id
        display.setdefault("action_type", pending.action_type)
        display.setdefault(
            "expires_at",
            pending.expires_at.isoformat(timespec="seconds") if pending.expires_at else None,
        )
        return display
