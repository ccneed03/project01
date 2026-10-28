"""
Chat Routes
"""
import json
from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify, Response, stream_with_context
from flask_login import login_required, current_user
from app.services.knowledge_service import KnowledgeService
from app.services.assistant_service import AssistantService
from app.extensions import db, csrf
from app.models.chat import ChatSession, ChatMessage
from app.services.retrieval_service import RetrievalService
from app.services.chat_service import ChatService

chat_bp = Blueprint("chat", __name__)


def _sse(event: str, data: dict) -> str:
    """Build a server-sent event payload."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _as_bool(value) -> bool:
    """Parse truthy values from JSON payloads."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@chat_bp.route("/knowledge-bases/<int:kb_id>/chat")
@login_required
def chat(kb_id):
    kb = KnowledgeService.get_knowledge_base(kb_id)
    if not kb:
        flash("知识库不存在", "danger")
        return redirect(url_for("admin.knowledge_bases"))

    sessions = ChatSession.query.filter_by(
        knowledge_base_id=kb_id,
        user_id=current_user.id,
    ).order_by(ChatSession.updated_at.desc()).limit(10).all()

    return render_template("chat/chat.html", kb=kb, sessions=sessions)


@chat_bp.route("/assistants/<int:assistant_id>/chat")
@login_required
def assistant_chat(assistant_id):
    assistant = AssistantService.get_assistant(assistant_id)
    if not assistant:
        flash("应用不存在", "danger")
        return redirect(url_for("admin.smart_assistants"))

    sessions = ChatSession.query.filter_by(
        assistant_id=assistant_id,
        user_id=current_user.id,
    ).order_by(ChatSession.updated_at.desc()).limit(20).all()

    return render_template(
        "assistants/chat.html",
        assistant=assistant,
        sessions=sessions,
    )


@chat_bp.route("/knowledge-bases/<int:kb_id>/chat/new", methods=["POST"])
@csrf.exempt
@login_required
def new_session(kb_id):
    kb = KnowledgeService.get_knowledge_base(kb_id)
    if not kb:
        return jsonify({"success": False, "message": "知识库不存在"}), 404

    session = ChatSession(
        knowledge_base_id=kb_id,
        user_id=current_user.id,
        title="新对话",
    )
    db.session.add(session)
    db.session.commit()

    return jsonify({
        "success": True,
        "session_id": session.id,
        "title": session.title,
    })


@chat_bp.route("/chat/<int:session_id>/send", methods=["POST"])
@csrf.exempt
@login_required
def send_message(session_id):
    session = db.session.get(ChatSession, session_id)
    if not session or session.user_id != current_user.id:
        return jsonify({"success": False, "message": "对话不存在"}), 404

    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"success": False, "message": "问题不能为空"}), 400

    user_msg = ChatMessage(
        session_id=session_id,
        role="user",
        content=query,
    )
    db.session.add(user_msg)

    retrieval = RetrievalService(session.knowledge_base)
    results = retrieval.search(query=query)

    context = "\n\n".join([r["text"] for r in results])

    response = ChatService.generate_response(query, context, session.knowledge_base)

    assistant_msg = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=response,
    )
    db.session.add(assistant_msg)

    session.updated_at = db.func.now()
    db.session.commit()

    return jsonify({
        "success": True,
        "response": response,
        "references": results,
    })


@chat_bp.route("/chat/<int:session_id>/stream", methods=["POST"])
@csrf.exempt
@login_required
def stream_message(session_id):
    session = db.session.get(ChatSession, session_id)
    if not session or session.user_id != current_user.id:
        return jsonify({"success": False, "message": "对话不存在"}), 404

    data = request.get_json() or {}
    query = data.get("query", "").strip()
    enable_thinking = _as_bool(data.get("enable_thinking"))

    if not query:
        return jsonify({"success": False, "message": "问题不能为空"}), 400

    user_msg = ChatMessage(session_id=session_id, role="user", content=query)
    db.session.add(user_msg)
    db.session.commit()

    chat_session_id = session.id
    knowledge_base_id = session.knowledge_base_id

    @stream_with_context
    def generate():
        response_parts: list[str] = []
        reasoning_parts: list[str] = []
        yield _sse("start", {"success": True, "session_id": chat_session_id})

        try:
            knowledge_base = KnowledgeService.get_knowledge_base(knowledge_base_id) if knowledge_base_id else None
            yield _sse("status", {"message": "正在检索知识库..."})
            retrieval = RetrievalService(knowledge_base)
            results = retrieval.search(query=query)
            context = "\n\n".join([r["text"] for r in results])
            yield _sse("references", {"references": results})
            yield _sse("status", {"message": enable_thinking and "正在思考并生成回答..." or "正在生成回答..."})

            for chunk in ChatService.stream_events(
                query=query,
                context=context,
                knowledge_base=knowledge_base,
                enable_thinking=enable_thinking,
            ):
                if chunk["type"] == "reasoning":
                    reasoning_parts.append(chunk["content"])
                    yield _sse("reasoning", {"content": chunk["content"]})
                    continue

                response_parts.append(chunk["content"])
                yield _sse("delta", {"content": chunk["content"]})

            final_response = "".join(response_parts).strip()
            current_session = db.session.get(ChatSession, chat_session_id)
            assistant_msg = ChatMessage(
                session_id=chat_session_id,
                role="assistant",
                content=final_response,
                reasoning_content="".join(reasoning_parts).strip() or None,
                references=results,
            )
            db.session.add(assistant_msg)
            if current_session:
                current_session.updated_at = db.func.now()
            db.session.commit()
            yield _sse("end", {"success": True, "response": final_response})
        except Exception as exc:
            db.session.rollback()
            yield _sse("error", {"success": False, "message": str(exc)})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@chat_bp.route("/chat/<int:session_id>/history")
@login_required
def chat_history(session_id):
    session = db.session.get(ChatSession, session_id)
    if not session or session.user_id != current_user.id:
        return jsonify({"success": False, "message": "对话不存在"}), 404

    messages = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.created_at).all()

    return jsonify({
        "success": True,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "reasoning_content": m.reasoning_content,
            }
            for m in messages
        ],
    })


@chat_bp.route("/chat/<int:session_id>/delete", methods=["POST"])
@csrf.exempt
@login_required
def delete_session(session_id):
    session = db.session.get(ChatSession, session_id)
    if not session or session.user_id != current_user.id:
        return jsonify({"success": False, "message": "对话不存在"}), 404

    db.session.delete(session)
    db.session.commit()
    return jsonify({"success": True})
