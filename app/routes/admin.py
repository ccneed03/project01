"""
Admin Routes
"""
from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user
from app.utils import admin_required
from app.services.knowledge_service import KnowledgeService
from app.services.model_catalog_service import ModelCatalogService
from app.services.assistant_service import AssistantService

admin_bp = Blueprint("admin", __name__)


@admin_bp.route("/")
@login_required
def index():
    from app.models.knowledge_base import KnowledgeBase
    from app.models.document import Document
    from app.models.user import User
    from app.models.assistant import Assistant

    kb_count = KnowledgeBase.query.filter_by(is_deleted=False).count()
    doc_count = Document.query.filter_by(is_deleted=False).count()
    user_count = User.query.count()
    assistant_count = Assistant.query.filter_by(is_deleted=False).count()

    return render_template(
        "admin/index.html",
        kb_count=kb_count,
        doc_count=doc_count,
        user_count=user_count,
        assistant_count=assistant_count,
    )


@admin_bp.route("/knowledge-bases")
@login_required
def knowledge_bases():
    page = request.args.get("page", 1, type=int)
    per_page = 20
    search = request.args.get("search", "")

    kbs, total = KnowledgeService.list_knowledge_bases(page, per_page, search if search else None)

    return render_template(
        "admin/knowledge_bases.html",
        knowledge_bases=kbs,
        total=total,
        page=page,
        per_page=per_page,
        search=search,
    )


@admin_bp.route("/knowledge-bases/create", methods=["GET", "POST"])
@login_required
def create_knowledge_base():
    embedding_models = ModelCatalogService.list_embedding_models()

    if request.method == "POST":
        name = request.form.get("name")
        description = request.form.get("description", "")
        default_embedding_model = request.form.get("default_embedding_model", "").strip() or None
        split_separator = request.form.get("split_separator", "\n\n")
        chunk_size = request.form.get("chunk_size", 500, type=int)
        chunk_overlap = request.form.get("chunk_overlap", 50, type=int)
        retrieval_method = request.form.get("retrieval_method", "hybrid")
        rerank_k = request.form.get("rerank_k", 6, type=int)
        enable_rerank = request.form.get("enable_rerank", "on") == "on"

        success, message, kb = KnowledgeService.create_knowledge_base(
            name=name,
            description=description,
            created_by=current_user.id,
            default_embedding_model=default_embedding_model,
            split_separator=split_separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            retrieval_method=retrieval_method,
            rerank_k=rerank_k,
            enable_rerank=enable_rerank,
        )

        if success:
            flash(f"知识库 '{name}' 创建成功", "success")
            return redirect(url_for("admin.knowledge_bases"))
        else:
            flash(message, "danger")

    return render_template(
        "admin/create_knowledge_base.html",
        embedding_models=embedding_models,
    )


@admin_bp.route("/knowledge-bases/<int:kb_id>/edit", methods=["GET", "POST"])
@login_required
def edit_knowledge_base(kb_id):
    kb = KnowledgeService.get_knowledge_base(kb_id)
    if not kb:
        flash("知识库不存在", "danger")
        return redirect(url_for("admin.knowledge_bases"))

    embedding_models = ModelCatalogService.list_embedding_models()

    if request.method == "POST":
        name = request.form.get("name")
        description = request.form.get("description")
        default_embedding_model = request.form.get("default_embedding_model", "").strip() or None
        split_separator = request.form.get("split_separator")
        chunk_size = request.form.get("chunk_size", type=int)
        chunk_overlap = request.form.get("chunk_overlap", type=int)
        retrieval_method = request.form.get("retrieval_method")
        rerank_k = request.form.get("rerank_k", type=int)
        enable_rerank = request.form.get("enable_rerank") == "on"

        success, message, kb = KnowledgeService.update_knowledge_base(
            kb_id=kb_id,
            name=name,
            description=description,
            default_embedding_model=default_embedding_model,
            split_separator=split_separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            retrieval_method=retrieval_method,
            rerank_k=rerank_k,
            enable_rerank=enable_rerank,
        )

        if success:
            flash("知识库更新成功", "success")
            return redirect(url_for("admin.knowledge_bases"))
        else:
            flash(message, "danger")

    return render_template(
        "admin/edit_knowledge_base.html",
        kb=kb,
        embedding_models=embedding_models,
    )


@admin_bp.route("/knowledge-bases/<int:kb_id>/delete", methods=["POST"])
@login_required
def delete_knowledge_base(kb_id):
    success, message = KnowledgeService.delete_knowledge_base(kb_id)
    if success:
        flash("知识库已删除", "success")
    else:
        flash(message, "danger")
    return redirect(url_for("admin.knowledge_bases"))


@admin_bp.route("/users")
@login_required
@admin_required
def users():
    from app.models.user import User
    page = request.args.get("page", 1, type=int)
    per_page = 20

    query = User.query
    total = query.count()
    users = query.order_by(User.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    ).items

    return render_template("admin/users.html", users=users, total=total, page=page, per_page=per_page)


@admin_bp.route("/assistant")
@admin_bp.route("/workbench")
@login_required
def assistant_workbench():
    page = request.args.get("page", 1, type=int)
    per_page = 12
    search = request.args.get("search", "")
    assistants, total = AssistantService.list_assistants(
        page=page,
        per_page=per_page,
        search=search or None,
    )

    return render_template(
        "admin/assistant_workbench.html",
        assistants=assistants,
        total=total,
        page=page,
        per_page=per_page,
        search=search,
    )


@admin_bp.route("/workbench/assistants/create", methods=["GET", "POST"])
@login_required
def create_assistant():
    knowledge_bases, _ = KnowledgeService.list_knowledge_bases(page=1, per_page=200)
    rerank_models = ModelCatalogService.list_rerank_models()
    llm_models = AssistantService.list_llm_models()

    if request.method == "POST":
        name = request.form.get("name", "")
        description = request.form.get("description", "")
        llm_model = request.form.get("llm_model", "")
        system_prompt = request.form.get("system_prompt", "")
        opening_message = request.form.get("opening_message", "")
        knowledge_base_id = request.form.get("knowledge_base_id", type=int)
        retrieval_mode = request.form.get("retrieval_mode", "hybrid")
        top_k = request.form.get("top_k", 10, type=int)
        enable_rerank = request.form.get("enable_rerank") == "on"
        rerank_model = request.form.get("rerank_model", "")
        score_threshold_raw = request.form.get("score_threshold", "").strip()
        try:
            score_threshold = float(score_threshold_raw) if score_threshold_raw else None
        except ValueError:
            score_threshold = None

        success, message, assistant = AssistantService.create_assistant(
            name=name,
            description=description,
            llm_model=llm_model,
            system_prompt=system_prompt,
            opening_message=opening_message,
            knowledge_base_id=knowledge_base_id,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            enable_rerank=enable_rerank,
            rerank_model=rerank_model,
            score_threshold=score_threshold,
            created_by=current_user.id,
        )
        if success:
            flash(f"应用 '{assistant.name}' 创建成功", "success")
            return redirect(url_for("admin.assistant_workbench"))
        flash(message, "danger")

    return render_template(
        "admin/assistant_form.html",
        mode="create",
        assistant=None,
        knowledge_bases=knowledge_bases,
        rerank_models=rerank_models,
        llm_models=llm_models,
    )


@admin_bp.route("/workbench/assistants/<int:assistant_id>/edit", methods=["GET", "POST"])
@login_required
def edit_assistant(assistant_id):
    assistant = AssistantService.get_assistant(assistant_id)
    if not assistant:
        flash("应用不存在", "danger")
        return redirect(url_for("admin.assistant_workbench"))

    knowledge_bases, _ = KnowledgeService.list_knowledge_bases(page=1, per_page=200)
    rerank_models = ModelCatalogService.list_rerank_models()
    llm_models = AssistantService.list_llm_models()

    if request.method == "POST":
        score_threshold_raw = request.form.get("score_threshold", "").strip()
        try:
            score_threshold = float(score_threshold_raw) if score_threshold_raw else None
        except ValueError:
            score_threshold = None

        success, message, _ = AssistantService.update_assistant(
            assistant_id,
            name=request.form.get("name", ""),
            description=request.form.get("description", ""),
            llm_model=request.form.get("llm_model", ""),
            system_prompt=request.form.get("system_prompt", ""),
            opening_message=request.form.get("opening_message", ""),
            knowledge_base_id=request.form.get("knowledge_base_id", type=int),
            retrieval_mode=request.form.get("retrieval_mode", "hybrid"),
            top_k=request.form.get("top_k", 10, type=int),
            enable_rerank=request.form.get("enable_rerank") == "on",
            rerank_model=request.form.get("rerank_model", ""),
            score_threshold=score_threshold,
            status=request.form.get("status", "published"),
        )
        if success:
            flash("应用更新成功", "success")
            return redirect(url_for("admin.assistant_workbench"))
        flash(message, "danger")

    return render_template(
        "admin/assistant_form.html",
        mode="edit",
        assistant=assistant,
        knowledge_bases=knowledge_bases,
        rerank_models=rerank_models,
        llm_models=llm_models,
    )


@admin_bp.route("/workbench/assistants/<int:assistant_id>/delete", methods=["POST"])
@login_required
def delete_assistant(assistant_id):
    success, message = AssistantService.delete_assistant(assistant_id)
    flash(message, "success" if success else "danger")
    return redirect(url_for("admin.assistant_workbench"))


@admin_bp.route("/smart-assistants")
@login_required
def smart_assistants():
    page = request.args.get("page", 1, type=int)
    per_page = 12
    search = request.args.get("search", "")
    assistants, total = AssistantService.list_assistants(
        page=page,
        per_page=per_page,
        search=search or None,
    )
    return render_template(
        "admin/assistant_gallery.html",
        assistants=assistants,
        total=total,
        page=page,
        per_page=per_page,
        search=search,
    )
