"""
API Routes
"""
import json
from flask import Blueprint, jsonify, request, Response, stream_with_context
from flask_login import login_required, current_user
from app.services.knowledge_service import KnowledgeService
from app.services.assistant_service import AssistantService
from app.services.retrieval_service import RetrievalService
from app.services.model_catalog_service import ModelCatalogService
from app.services.document_index_service import DocumentIndexService
from app.services.document_service import DocumentService
from app.services.chat_service import ChatService
from app.extensions import db, csrf

api_bp = Blueprint("api", __name__)


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


@api_bp.route("/health")
def health():
    return jsonify({"status": "ok"})


@api_bp.route("/embedding-models")
@login_required
def embedding_models():
    return jsonify({"success": True, "models": ModelCatalogService.list_embedding_models()})


@api_bp.route("/rerank-models")
@login_required
def rerank_models():
    return jsonify({"success": True, "models": ModelCatalogService.list_rerank_models()})


@api_bp.route("/llm-models")
@login_required
def llm_models():
    return jsonify({"success": True, "models": AssistantService.list_llm_models()})


@api_bp.route("/retrieval/test", methods=["POST"])
@csrf.exempt
@login_required
def retrieval_test():
    data = request.get_json() or {}
    knowledge_base_id = data.get("knowledge_base_id")
    query = (data.get("query") or "").strip()
    retrieval_mode = data.get("retrieval_mode", "hybrid")
    top_k = data.get("top_k", 10)
    enable_rerank = _as_bool(data.get("enable_rerank", True))
    rerank_model = data.get("rerank_model")
    score_threshold = data.get("score_threshold")

    if not knowledge_base_id:
        return jsonify({"success": False, "message": "知识库不能为空"}), 400
    if not query:
        return jsonify({"success": False, "message": "查询不能为空"}), 400

    kb = KnowledgeService.get_knowledge_base(int(knowledge_base_id))
    if not kb:
        return jsonify({"success": False, "message": "知识库不存在"}), 404

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = 10

    try:
        score_threshold = None if score_threshold in ("", None) else float(score_threshold)
    except (TypeError, ValueError):
        score_threshold = None

    retrieval = RetrievalService(kb)
    results = retrieval.search(
        query=query,
        method=retrieval_mode,
        top_k=top_k,
        enable_rerank=enable_rerank,
        rerank_model=rerank_model,
        score_threshold=score_threshold,
    )

    return jsonify(
        {
            "success": True,
            "knowledge_base_id": kb.id,
            "query": query,
            "retrieval_mode": retrieval_mode,
            "top_k": top_k,
            "enable_rerank": enable_rerank,
            "rerank_model": rerank_model,
            "score_threshold": score_threshold,
            "results": results,
            "total": len(results),
        }
    )


@api_bp.route("/index-tasks/<int:task_id>")
@login_required
def index_task_detail(task_id):
    task = DocumentIndexService.get_task(task_id)
    if not task:
        return jsonify({"success": False, "message": "任务不存在"}), 404
    return jsonify({"success": True, "task": DocumentService.serialize_index_task(task)})


@api_bp.route("/documents/<int:document_id>/preview-result")
@login_required
def preview_result(document_id):
    result = DocumentIndexService.get_preview_result(document_id)
    if result is None:
        return jsonify({"success": False, "message": "预览结果不存在"}), 404
    return jsonify({"success": True, "document_id": document_id, "preview": result})


@api_bp.route("/knowledge-bases/<int:kb_id>/retrieve", methods=["POST"])
@csrf.exempt
@login_required
def retrieve(kb_id):
    kb = KnowledgeService.get_knowledge_base(kb_id)
    if not kb:
        return jsonify({"success": False, "message": "知识库不存在"}), 404

    data = request.get_json()
    query = data.get("query", "").strip()
    method = data.get("method", "hybrid")
    top_k = data.get("top_k", kb.rerank_k or 6)

    if not query:
        return jsonify({"success": False, "message": "查询不能为空"}), 400

    retrieval = RetrievalService(kb)
    results = retrieval.search(
        query=query,
        method=method,
        top_k=top_k,
        enable_rerank=kb.enable_rerank,
    )

    return jsonify({
        "success": True,
        "query": query,
        "method": method,
        "results": results,
        "total": len(results),
    })


@api_bp.route("/knowledge-bases/<int:kb_id>/chat", methods=["POST"])
@csrf.exempt
@login_required
def chat(kb_id):
    kb = KnowledgeService.get_knowledge_base(kb_id)
    if not kb:
        return jsonify({"success": False, "message": "知识库不存在"}), 404

    data = request.get_json()
    query = data.get("query", "").strip()
    session_id = data.get("session_id")

    if not query:
        return jsonify({"success": False, "message": "问题不能为空"}), 400

    from app.services.retrieval_service import RetrievalService
    from app.models.chat import ChatSession, ChatMessage

    # Get or create session
    if session_id:
        session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first()
    else:
        session = ChatSession(
            knowledge_base_id=kb_id,
            user_id=current_user.id,
            title=query[:50],
        )
        db.session.add(session)
        db.session.commit()

    # Save user message
    user_msg = ChatMessage(session_id=session.id, role="user", content=query)
    db.session.add(user_msg)

    # Retrieve context
    retrieval = RetrievalService(kb)
    results = retrieval.search(query=query)
    context = "\n\n".join([r["text"] for r in results])

    # Generate response
    response = ChatService.generate_response(query, context, kb)

    # Save assistant message
    assistant_msg = ChatMessage(session_id=session.id, role="assistant", content=response)
    db.session.add(assistant_msg)
    db.session.commit()

    return jsonify({
        "success": True,
        "session_id": session.id,
        "response": response,
        "references": results,
    })


@api_bp.route("/knowledge-bases/<int:kb_id>/chat/stream", methods=["POST"])
@csrf.exempt
@login_required
def chat_stream(kb_id):
    kb = KnowledgeService.get_knowledge_base(kb_id)
    if not kb:
        return jsonify({"success": False, "message": "知识库不存在"}), 404

    data = request.get_json() or {}
    query = data.get("query", "").strip()
    session_id = data.get("session_id")
    enable_thinking = _as_bool(data.get("enable_thinking"))

    if not query:
        return jsonify({"success": False, "message": "问题不能为空"}), 400

    from app.models.chat import ChatSession, ChatMessage

    if session_id:
        session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first()
        if not session:
            return jsonify({"success": False, "message": "对话不存在"}), 404
    else:
        session = ChatSession(
            knowledge_base_id=kb_id,
            user_id=current_user.id,
            title=query[:50],
        )
        db.session.add(session)
        db.session.commit()

    user_msg = ChatMessage(session_id=session.id, role="user", content=query)
    db.session.add(user_msg)
    db.session.commit()

    chat_session_id = session.id

    @stream_with_context
    def generate():
        response_parts: list[str] = []
        reasoning_parts: list[str] = []
        yield _sse("start", {"success": True, "session_id": chat_session_id})

        try:
            yield _sse("status", {"message": "正在检索知识库..."})
            retrieval = RetrievalService(kb)
            results = retrieval.search(query=query)
            context = "\n\n".join([r["text"] for r in results])
            yield _sse("references", {"references": results})
            yield _sse("status", {"message": "正在生成回答..."})

            for chunk in ChatService.stream_events(
                query=query,
                context=context,
                knowledge_base=kb,
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
            yield _sse("end", {"success": True, "response": final_response, "session_id": chat_session_id})
        except Exception as exc:
            db.session.rollback()
            yield _sse("error", {"success": False, "message": str(exc), "session_id": chat_session_id})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@api_bp.route("/assistants/<int:assistant_id>/chat/stream", methods=["POST"])
@csrf.exempt
@login_required
def assistant_chat_stream(assistant_id):
    assistant = AssistantService.get_assistant(assistant_id)
    if not assistant:
        return jsonify({"success": False, "message": "应用不存在"}), 404

    data = request.get_json() or {}
    query = (data.get("query") or "").strip()
    session_id = data.get("session_id")
    enable_thinking = _as_bool(data.get("enable_thinking"))

    if not query:
        return jsonify({"success": False, "message": "问题不能为空"}), 400

    from app.models.chat import ChatSession, ChatMessage

    kb_binding = assistant.primary_knowledge_base_config
    kb = kb_binding.knowledge_base if kb_binding else None
    assistant_entity_id = assistant.id

    if session_id:
        session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first()
        if not session:
            return jsonify({"success": False, "message": "对话不存在"}), 404
        if session.assistant_id != assistant.id:
            return jsonify({"success": False, "message": "当前对话与所选应用不一致，请新建对话"}), 400
    else:
        session = ChatSession(
            assistant_id=assistant.id,
            knowledge_base_id=kb.id if kb else None,
            user_id=current_user.id,
            title=query[:50],
        )
        db.session.add(session)
        db.session.commit()

    user_msg = ChatMessage(session_id=session.id, role="user", content=query)
    db.session.add(user_msg)
    db.session.commit()

    chat_session_id = session.id
    knowledge_base_id = kb.id if kb else None

    @stream_with_context
    def generate():
        response_parts: list[str] = []
        reasoning_parts: list[str] = []
        yield _sse(
            "start",
            {
                "success": True,
                "session_id": chat_session_id,
                "assistant_id": assistant_entity_id,
                "knowledge_base_id": knowledge_base_id,
            },
        )

        try:
            results = []
            context = ""
            active_assistant = AssistantService.get_assistant(assistant_id)
            active_binding = active_assistant.primary_knowledge_base_config if active_assistant else None
            active_kb = active_binding.knowledge_base if active_binding else None

            if active_kb and active_binding:
                yield _sse("status", {"message": "正在检索知识库..."})
                retrieval = RetrievalService(active_kb)
                results = retrieval.search(
                    query=query,
                    method=active_binding.retrieval_mode,
                    top_k=active_binding.top_k,
                    enable_rerank=active_binding.enable_rerank,
                    rerank_model=active_binding.rerank_model,
                    score_threshold=active_binding.score_threshold,
                )
                context = "\n\n".join([r["text"] for r in results])
                yield _sse("references", {"references": results})

            yield _sse("status", {"message": enable_thinking and "正在思考并生成回答..." or "正在生成回答..."})
            for chunk in ChatService.stream_events(
                query=query,
                context=context,
                knowledge_base=active_kb,
                assistant=active_assistant,
                model=active_assistant.llm_model if active_assistant else None,
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
            yield _sse(
                "end",
                {
                    "success": True,
                    "response": final_response,
                    "session_id": chat_session_id,
                    "assistant_id": assistant_entity_id,
                    "knowledge_base_id": knowledge_base_id,
                },
            )
        except Exception as exc:
            db.session.rollback()
            yield _sse(
                "error",
                {
                    "success": False,
                    "message": str(exc),
                    "session_id": chat_session_id,
                    "assistant_id": assistant_entity_id,
                    "knowledge_base_id": knowledge_base_id,
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
