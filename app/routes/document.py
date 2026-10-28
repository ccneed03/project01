"""
Document Routes
"""
from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify
from flask_login import login_required, current_user
from app.models.chunk import Chunk
from app.models.document_index_config import DocumentIndexConfig
from app.models.document_index_task import DocumentIndexTask
from app.services.chunk_service import ChunkService
from app.services.document_index_service import DocumentIndexService
from app.services.document_service import DocumentService
from app.services.knowledge_service import KnowledgeService
from app.extensions import db

document_bp = Blueprint("document", __name__)


@document_bp.route("/knowledge-bases/<int:kb_id>/documents")
@login_required
def list_documents(kb_id):
    kb = KnowledgeService.get_knowledge_base(kb_id)
    if not kb:
        flash("知识库不存在", "danger")
        return redirect(url_for("admin.knowledge_bases"))

    page = request.args.get("page", 1, type=int)
    per_page = 20
    status = request.args.get("status")
    search = request.args.get("search")

    docs, total = DocumentService.list_documents(
        knowledge_base_id=kb_id,
        page=page,
        per_page=per_page,
        status=status,
        search=search,
    )

    return render_template(
        "documents/list.html",
        kb=kb,
        documents=docs,
        total=total,
        page=page,
        per_page=per_page,
        status=status,
        search=search,
    )


@document_bp.route("/documents")
@login_required
def all_documents():
    """List documents across all knowledge bases (dashboard entry point)."""
    page = request.args.get("page", 1, type=int)
    per_page = 20
    status = request.args.get("status")
    search = request.args.get("search")

    docs, total = DocumentService.list_documents(
        knowledge_base_id=None,
        page=page,
        per_page=per_page,
        status=status,
        search=search,
    )

    return render_template(
        "documents/all.html",
        documents=docs,
        total=total,
        page=page,
        per_page=per_page,
        status=status,
        search=search,
    )


@document_bp.route("/api/knowledge-bases/<int:kb_id>/documents/status")
@login_required
def document_status_list(kb_id):
    kb = KnowledgeService.get_knowledge_base(kb_id)
    if not kb:
        return jsonify({"success": False, "message": "知识库不存在"}), 404

    status = request.args.get("status")
    search = request.args.get("search")

    docs, total = DocumentService.list_documents(
        knowledge_base_id=kb_id,
        page=1,
        per_page=100,
        status=status,
        search=search,
    )

    return jsonify(
        {
            "success": True,
            "documents": [DocumentService.serialize_document(doc) for doc in docs],
            "total": total,
            "has_active_tasks": any(doc.status in DocumentService.ACTIVE_STATUSES for doc in docs),
        }
    )


@document_bp.route("/knowledge-bases/<int:kb_id>/documents/upload", methods=["GET", "POST"])
@login_required
def upload_document(kb_id):
    kb = KnowledgeService.get_knowledge_base(kb_id)
    if not kb:
        flash("知识库不存在", "danger")
        return redirect(url_for("admin.knowledge_bases"))

    if request.method == "POST":
        if "file" not in request.files:
            flash("没有选择文件", "warning")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("没有选择文件", "warning")
            return redirect(request.url)

        title = request.form.get("title", "").strip() or None
        chunk_mode = request.form.get("chunk_mode", "paragraph").strip() or "paragraph"
        chunk_size_str = request.form.get("chunk_size", "")
        chunk_size = int(chunk_size_str) if chunk_size_str else 500
        chunk_overlap_str = request.form.get("chunk_overlap", "")
        chunk_overlap = int(chunk_overlap_str) if chunk_overlap_str else 50
        split_separator = request.form.get("split_separator", "\n\n")

        success, message, doc = DocumentService.create_document(
            knowledge_base_id=kb_id,
            file=file,
            created_by=current_user.id,
        )

        if success:
            doc.title = title
            doc.status = "uploaded"
            embedding_model = (kb.default_embedding_model or "").strip()
            if not embedding_model:
                flash("当前知识库还没有配置默认嵌入模型，请先编辑知识库后再上传文档。", "danger")
                return redirect(url_for("admin.edit_knowledge_base", kb_id=kb.id))

            config_success, config_message, config = DocumentIndexService.create_index_config(
                document_id=doc.id,
                created_by=current_user.id,
                embedding_model=embedding_model,
                chunk_mode=chunk_mode,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                split_separator=split_separator,
            )
            if not config_success or not config:
                flash(config_message, "danger")
                return redirect(request.url)

            task_success, task_message, task = DocumentIndexService.create_preview_task(
                document_id=doc.id,
                config_id=config.id,
                created_by=current_user.id,
            )
            if not task_success or not task:
                flash(task_message, "danger")
                return redirect(request.url)

            from app.tasks import preview_document_task

            try:
                preview_document_task.delay(task.id)
                flash(f"文档 '{doc.original_filename}' 上传成功，正在生成预览...", "success")
            except Exception:
                process_success, process_message = DocumentIndexService.run_preview_task(task.id)
                flash_category = "success" if process_success else "warning"
                flash(
                    f"文档 '{doc.original_filename}' 已上传，但后台队列不可用，改为同步预览：{process_message}",
                    flash_category,
                )
            return redirect(url_for("document.preview_document", doc_id=doc.id))
        else:
            flash(message, "danger")

    return render_template(
        "documents/upload.html",
        kb=kb,
    )


@document_bp.route("/documents/<int:doc_id>")
@login_required
def view_document(doc_id):
    doc = DocumentService.get_document(doc_id)
    if not doc:
        flash("文档不存在", "danger")
        return redirect(url_for("admin.knowledge_bases"))

    current_config = db.session.get(DocumentIndexConfig, doc.current_config_id) if doc.current_config_id else None
    current_task = db.session.get(DocumentIndexTask, doc.current_index_task_id) if doc.current_index_task_id else None
    return render_template("documents/view.html", doc=doc, current_config=current_config, current_task=current_task)


@document_bp.route("/documents/<int:doc_id>/preview")
@login_required
def preview_document(doc_id):
    doc = DocumentService.get_document(doc_id)
    if not doc:
        flash("文档不存在", "danger")
        return redirect(url_for("admin.knowledge_bases"))

    config = db.session.get(DocumentIndexConfig, doc.current_config_id) if doc.current_config_id else None
    preview_task = (
        DocumentIndexTask.query.filter_by(
            document_id=doc.id,
            task_type=DocumentIndexService.PREVIEW_TASK,
        )
        .order_by(DocumentIndexTask.created_at.desc())
        .first()
    )
    current_task = db.session.get(DocumentIndexTask, doc.current_index_task_id) if doc.current_index_task_id else None
    preview_result = DocumentIndexService.get_preview_result(doc.id)
    return render_template(
        "documents/preview.html",
        doc=doc,
        config=config,
        preview_task=preview_task,
        current_task=current_task,
        preview_result=preview_result,
    )


@document_bp.route("/documents/<int:doc_id>/chunks")
@login_required
def view_document_chunks(doc_id):
    doc = DocumentService.get_document(doc_id)
    if not doc:
        flash("文档不存在", "danger")
        return redirect(url_for("admin.knowledge_bases"))

    page = request.args.get("page", 1, type=int)
    per_page = 50

    chunks, total = DocumentService.get_document_chunks(doc_id, page, per_page)

    return render_template(
        "documents/chunks.html",
        doc=doc,
        chunks=chunks,
        total=total,
        page=page,
        per_page=per_page,
    )


@document_bp.route("/chunks/<int:chunk_id>", methods=["GET", "POST"])
@login_required
def chunk_detail(chunk_id):
    chunk = ChunkService.get_chunk(chunk_id)
    if not chunk:
        flash("Chunk 不存在", "danger")
        return redirect(url_for("admin.knowledge_bases"))

    if request.method == "POST":
        content = request.form.get("content", "")
        tags = request.form.get("tags", "")
        note = request.form.get("note", "")

        success, message, _ = ChunkService.update_chunk(
            chunk_id=chunk_id,
            content=content,
            tags_text=tags,
            note=note,
        )
        flash(message, "success" if success else "danger")
        return redirect(url_for("document.chunk_detail", chunk_id=chunk_id))

    return render_template("documents/chunk_detail.html", chunk=chunk)


@document_bp.route("/documents/<int:doc_id>/delete", methods=["POST"])
@login_required
def delete_document(doc_id):
    success, message = DocumentService.delete_document(doc_id)
    if success:
        flash("文档已删除", "success")
    else:
        flash(message, "danger")
    return redirect(url_for("document.list_documents", kb_id=request.form.get("kb_id")))


@document_bp.route("/documents/<int:doc_id>/index", methods=["POST"])
@login_required
def confirm_document_index(doc_id):
    doc = DocumentService.get_document(doc_id)
    if not doc:
        flash("文档不存在", "danger")
        return redirect(url_for("admin.knowledge_bases"))

    config_id = request.form.get("config_id", type=int) or doc.current_config_id
    if not config_id:
        flash("索引配置不存在", "danger")
        return redirect(url_for("document.preview_document", doc_id=doc.id))

    success, message, task = DocumentIndexService.create_index_task(
        document_id=doc.id,
        config_id=config_id,
        created_by=current_user.id,
    )
    if not success or not task:
        flash(message, "danger")
        return redirect(url_for("document.preview_document", doc_id=doc.id))

    from app.tasks import index_document_task

    try:
        index_document_task.delay(task.id)
        flash("已创建正式索引任务，正在后台构建索引。", "success")
    except Exception:
        process_success, process_message = DocumentIndexService.run_index_task(task.id)
        flash(process_message, "success" if process_success else "warning")

    return redirect(url_for("document.preview_document", doc_id=doc.id))
