"""Agent streaming and confirmation endpoints."""
from __future__ import annotations

import json
from typing import Any

from flask import Blueprint, Response, jsonify, request, stream_with_context
from flask_login import current_user, login_required

from app.agents.gateway import ChatGateway
from app.agents.runtime import AgentRuntime
from app.extensions import csrf, db
from app.models.chat import ChatSession
from app.services.agent_service import AgentActionError, AgentService
from app.services.assistant_service import AssistantService
from app.tools.executor import ToolExecutor
from app.tools.registry import get_default_registry

agent_bp = Blueprint("agent", __name__)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@agent_bp.route("/assistants/<int:assistant_id>/agent/stream", methods=["POST"])
@csrf.exempt
@login_required
def agent_stream(assistant_id):
    """Run the agent and stream SSE events.

    The stream itself is exempt from CSRF, matching the existing chat
    endpoints.  The confirm/cancel endpoints remain protected.
    """
    assistant = AssistantService.get_assistant(assistant_id)
    if not assistant:
        return jsonify({"success": False, "message": "应用不存在"}), 404

    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    session_id = data.get("session_id")
    enable_thinking = _as_bool(data.get("enable_thinking"))

    if not query:
        return jsonify({"success": False, "message": "问题不能为空"}), 400

    binding = assistant.primary_knowledge_base_config
    knowledge_base = binding.knowledge_base if binding else None

    if session_id:
        session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first()
        if not session:
            return jsonify({"success": False, "message": "对话不存在"}), 404
        if session.assistant_id != assistant.id:
            return jsonify({"success": False, "message": "当前对话与所选应用不一致，请新建对话"}), 400
    else:
        session = ChatSession(
            assistant_id=assistant.id,
            knowledge_base_id=knowledge_base.id if knowledge_base else None,
            user_id=current_user.id,
            title=query[:50],
        )
        db.session.add(session)
        db.session.commit()

    user = current_user._get_current_object()
    run, context, state = AgentService.start_run(
        user=user,
        session=session,
        assistant=assistant,
        query=query,
    )

    gateway = ChatGateway(
        model=assistant.llm_model,
        enable_thinking=enable_thinking,
    )
    registry = get_default_registry()
    runtime = AgentRuntime(
        gateway=gateway,
        executor=ToolExecutor(registry),
        registry=registry,
    )

    chat_session_id = session.id
    run_id = run.run_id
    assistant_id_value = assistant.id

    @stream_with_context
    def generate():
        references: list[dict[str, Any]] = []
        yield _sse(
            "start",
            {
                "success": True,
                "session_id": chat_session_id,
                "assistant_id": assistant_id_value,
                "run_id": run_id,
            },
        )

        try:
            for event in runtime.run(state, context):
                event_type = event.get("type")

                if event_type == "reasoning":
                    yield _sse("reasoning", {"content": event.get("content") or ""})
                    continue

                if event_type == "delta":
                    yield _sse("delta", {"content": event.get("content") or ""})
                    continue

                if event_type == "tool_start":
                    yield _sse(
                        "tool_start",
                        {
                            "name": event.get("name"),
                            "label": event.get("label"),
                            "discard_content": bool(event.get("discard_content")),
                        },
                    )
                    continue

                if event_type == "tool_result":
                    AgentService.record_step(
                        run_id=run_id,
                        step_index=int(event.get("step_index") or state.step_count),
                        tool_name=event.get("name") or "unknown",
                        tool_call_id=event.get("tool_call_id"),
                        arguments=event.get("arguments") or {},
                        result_code=event.get("code") or "",
                        ok=bool(event.get("ok")),
                        duration_ms=int(event.get("duration_ms") or 0),
                    )
                    AgentService.save_state_by_run_id(run_id, state)

                    sources = event.get("sources") or []
                    if sources:
                        references = sources
                        yield _sse("references", {"references": sources})

                    yield _sse(
                        "tool_result",
                        {
                            "name": event.get("name"),
                            "ok": bool(event.get("ok")),
                            "code": event.get("code"),
                            "display_text": event.get("display_text") or "",
                            "duration_ms": int(event.get("duration_ms") or 0),
                        },
                    )
                    continue

                if event_type == "confirmation_required":
                    AgentService.save_state_by_run_id(
                        run_id, state, status="awaiting_confirmation"
                    )
                    yield _sse(
                        "confirmation_required",
                        {
                            "action": event.get("action") or {},
                            "action_id": event.get("action_id"),
                        },
                    )
                    return

                if event_type == "end":
                    answer = (event.get("content") or "").strip()
                    AgentService.save_state_by_run_id(run_id, state, status="completed")
                    AgentService.save_assistant_message(
                        session_id=chat_session_id,
                        content=answer,
                        references=references or None,
                    )
                    yield _sse(
                        "end",
                        {
                            "success": True,
                            "response": answer,
                            "session_id": chat_session_id,
                            "run_id": run_id,
                        },
                    )
                    return

                if event_type == "error":
                    AgentService.save_state_by_run_id(run_id, state, status="failed")
                    yield _sse(
                        "error",
                        {
                            "success": False,
                            "message": event.get("message") or "处理失败",
                            "session_id": chat_session_id,
                            "run_id": run_id,
                        },
                    )
                    return
        except Exception as exc:
            db.session.rollback()
            try:
                AgentService.mark_run_failed_by_id(run_id, str(exc))
            except Exception:
                db.session.rollback()
            yield _sse(
                "error",
                {
                    "success": False,
                    "message": f"处理失败：{exc}",
                    "session_id": chat_session_id,
                    "run_id": run_id,
                },
            )

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@agent_bp.route("/agent-actions/<int:action_id>/confirm", methods=["POST"])
@login_required
def confirm_action(action_id):
    try:
        result = AgentService.confirm_action(action_id, current_user.id)
        return jsonify(result)
    except AgentActionError as exc:
        return (
            jsonify({"success": False, "code": exc.code, "message": exc.message}),
            exc.http_status,
        )


@agent_bp.route("/agent-actions/<int:action_id>/cancel", methods=["POST"])
@login_required
def cancel_action(action_id):
    try:
        result = AgentService.cancel_action(action_id, current_user.id)
        return jsonify(result)
    except AgentActionError as exc:
        return (
            jsonify({"success": False, "code": exc.code, "message": exc.message}),
            exc.http_status,
        )


@agent_bp.route("/agent-actions/pending", methods=["GET"])
@login_required
def pending_actions():
    session_id = request.args.get("session_id", type=int)
    if not session_id:
        return jsonify({"success": False, "message": "session_id 不能为空"}), 400
    session = db.session.get(ChatSession, session_id)
    if not session or session.user_id != current_user.id:
        return jsonify({"success": False, "message": "对话不存在"}), 404
    return jsonify(
        {
            "success": True,
            "actions": AgentService.pending_actions(session_id, current_user.id),
        }
    )
