"""
Search Routes
"""
from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from app.services.knowledge_service import KnowledgeService
from app.services.retrieval_service import RetrievalService
from app.services.model_catalog_service import ModelCatalogService
from app.models.search_history import SearchHistory
from app.extensions import db

search_bp = Blueprint("search", __name__)


@search_bp.route("/knowledge-bases/<int:kb_id>/search")
@login_required
def search(kb_id):
    kb = KnowledgeService.get_knowledge_base(kb_id)
    if not kb:
        return {"success": False, "message": "知识库不存在"}, 404

    query = request.args.get("q", "").strip()
    method = request.args.get("method", "hybrid")
    top_k = request.args.get("top_k", 10, type=int)

    if not query:
        return render_template("search/search.html", kb=kb, results=[], query="", method=method)

    # Perform search
    retrieval = RetrievalService(kb)
    results = retrieval.search(
        query=query,
        method=method,
        top_k=top_k,
        enable_rerank=kb.enable_rerank,
    )

    # Save search history
    history = SearchHistory(
        knowledge_base_id=kb_id,
        user_id=current_user.id,
        query=query,
        method=method,
        top_k=top_k,
        result_count=len(results),
    )
    db.session.add(history)
    db.session.commit()

    return render_template(
        "search/search.html",
        kb=kb,
        results=results,
        query=query,
        method=method,
        top_k=top_k,
    )


@search_bp.route("/api/knowledge-bases/<int:kb_id>/search")
@login_required
def api_search(kb_id):
    kb = KnowledgeService.get_knowledge_base(kb_id)
    if not kb:
        return jsonify({"success": False, "message": "知识库不存在"}), 404

    query = request.args.get("q", "").strip()
    method = request.args.get("method", "hybrid")
    top_k = request.args.get("top_k", 10, type=int)

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


@search_bp.route("/retrieval-playground")
@login_required
def retrieval_playground():
    page = request.args.get("page", 1, type=int)
    knowledge_bases, _ = KnowledgeService.list_knowledge_bases(page=page, per_page=200)
    return render_template(
        "search/retrieval_playground.html",
        knowledge_bases=knowledge_bases,
        rerank_models=ModelCatalogService.list_rerank_models(),
    )
