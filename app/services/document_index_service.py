"""
Document indexing workflow service for preview and final indexing.
"""
from datetime import datetime
from typing import Optional

from app.extensions import db
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.document_index_config import DocumentIndexConfig
from app.models.document_index_task import DocumentIndexTask
from app.models.knowledge_base import KnowledgeBase
from app.services.document_parser_service import DocumentParserService
from app.services.document_service import DocumentService
from app.services.model_catalog_service import ModelCatalogService


class DocumentIndexService:
    """Create configs/tasks and run preview/index workflows."""

    PREVIEW_TASK = "preview"
    INDEX_TASK = "index"
    TASK_PENDING = "pending"
    TASK_RUNNING = "running"
    TASK_COMPLETED = "completed"
    TASK_FAILED = "failed"

    @staticmethod
    def create_index_config(
        document_id: int,
        created_by: int,
        embedding_model: str | None = None,
        chunk_mode: str = "paragraph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        split_separator: str | None = None,
        parse_strategy: str = "auto",
        activate: bool = True,
    ) -> tuple[bool, str, Optional[DocumentIndexConfig]]:
        """Create a new indexing config for a document."""
        document = db.session.get(Document, document_id)
        if not document or document.is_deleted:
            return False, "文档不存在", None

        knowledge_base = db.session.get(KnowledgeBase, document.knowledge_base_id)
        embedding_model = (embedding_model or "").strip() or (knowledge_base.default_embedding_model if knowledge_base else "")
        embedding_model = (embedding_model or "").strip()
        if not embedding_model:
            return False, "知识库未配置默认 Embedding 模型", None

        resolved_embedding_model = ModelCatalogService.resolve_embedding_model(embedding_model)
        if not resolved_embedding_model:
            return False, "Embedding 模型不存在", None

        latest = (
            DocumentIndexConfig.query.filter_by(document_id=document_id)
            .order_by(DocumentIndexConfig.version.desc())
            .first()
        )
        next_version = 1 if latest is None else latest.version + 1

        if activate:
            DocumentIndexConfig.query.filter_by(document_id=document_id, is_active=True).update({"is_active": False})

        config = DocumentIndexConfig(
            document_id=document.id,
            knowledge_base_id=document.knowledge_base_id,
            embedding_model=embedding_model,
            chunk_mode=chunk_mode,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_separator=split_separator,
            parse_strategy=parse_strategy,
            version=next_version,
            is_active=activate,
            created_by=created_by,
        )
        db.session.add(config)
        db.session.flush()

        if activate:
            document.current_config_id = config.id
            document.status = "uploaded"

        db.session.commit()
        return True, "索引配置创建成功", config

    @staticmethod
    def create_preview_task(document_id: int, config_id: int, created_by: int) -> tuple[bool, str, Optional[DocumentIndexTask]]:
        """Create a preview task for a document/config pair."""
        return DocumentIndexService._create_task(document_id, config_id, created_by, DocumentIndexService.PREVIEW_TASK)

    @staticmethod
    def create_index_task(document_id: int, config_id: int, created_by: int) -> tuple[bool, str, Optional[DocumentIndexTask]]:
        """Create a formal indexing task for a document/config pair."""
        return DocumentIndexService._create_task(document_id, config_id, created_by, DocumentIndexService.INDEX_TASK)

    @staticmethod
    def _create_task(
        document_id: int,
        config_id: int,
        created_by: int,
        task_type: str,
    ) -> tuple[bool, str, Optional[DocumentIndexTask]]:
        """Persist a task record."""
        document = db.session.get(Document, document_id)
        config = db.session.get(DocumentIndexConfig, config_id)
        if not document or document.is_deleted:
            return False, "文档不存在", None
        if not config or config.document_id != document_id:
            return False, "索引配置不存在", None

        task = DocumentIndexTask(
            document_id=document.id,
            knowledge_base_id=document.knowledge_base_id,
            config_id=config.id,
            task_type=task_type,
            status=DocumentIndexService.TASK_PENDING,
            progress=0,
            created_by=created_by,
        )
        db.session.add(task)
        db.session.flush()

        if task_type == DocumentIndexService.PREVIEW_TASK:
            document.status = "previewing"
        else:
            document.status = "indexing"
            document.current_index_task_id = task.id

        db.session.commit()
        return True, "任务创建成功", task

    @staticmethod
    def run_preview_task(task_id: int) -> tuple[bool, str]:
        """Execute preview parsing and chunk generation."""
        task = db.session.get(DocumentIndexTask, task_id)
        if not task or task.task_type != DocumentIndexService.PREVIEW_TASK:
            return False, "预览任务不存在"

        document = db.session.get(Document, task.document_id)
        config = db.session.get(DocumentIndexConfig, task.config_id)
        if not document or not config:
            return False, "预览任务关联数据不存在"

        try:
            DocumentIndexService._set_task_status(task, DocumentIndexService.TASK_RUNNING, progress=10)
            preview = DocumentParserService.build_preview(
                document=document,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                split_separator=config.split_separator,
                chunk_mode=config.chunk_mode,
            )
            task.result_summary = preview
            task.progress = 100
            task.status = DocumentIndexService.TASK_COMPLETED
            task.finished_at = datetime.now()
            document.status = "ready_to_index"
            document.chunk_count = preview["chunk_count"]
            document.current_config_id = config.id
            document.error_message = None
            db.session.commit()
            return True, f"预览完成，共 {preview['chunk_count']} 个 chunks"
        except Exception as exc:
            db.session.rollback()
            return DocumentIndexService._mark_task_failed(task_id, str(exc), document_id=document.id)

    @staticmethod
    def run_index_task(task_id: int) -> tuple[bool, str]:
        """Execute the final chunk/vector indexing workflow."""
        task = db.session.get(DocumentIndexTask, task_id)
        if not task or task.task_type != DocumentIndexService.INDEX_TASK:
            return False, "索引任务不存在"

        document = db.session.get(Document, task.document_id)
        config = db.session.get(DocumentIndexConfig, task.config_id)
        knowledge_base = db.session.get(KnowledgeBase, task.knowledge_base_id)
        if not document or not config or not knowledge_base:
            return False, "索引任务关联数据不存在"

        resolved_embedding_model = ModelCatalogService.resolve_embedding_model(config.embedding_model)
        if not resolved_embedding_model:
            return DocumentIndexService._mark_task_failed(task_id, "Embedding 模型不存在", document_id=document.id)

        try:
            DocumentIndexService._set_task_status(task, DocumentIndexService.TASK_RUNNING, progress=5)

            preview = DocumentParserService.build_preview(
                document=document,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                split_separator=config.split_separator,
                chunk_mode=config.chunk_mode,
            )
            chunks = [item["content"] for item in preview["preview_chunks"]]
            if preview["chunk_count"] > len(chunks):
                # Rebuild with full chunk list when preview result was truncated.
                text = DocumentParserService.extract_text(document)
                chunks = DocumentParserService.split_text(
                    text=text,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    split_separator=config.split_separator,
                    chunk_mode=config.chunk_mode,
                )

            DocumentIndexService._set_task_status(task, DocumentIndexService.TASK_RUNNING, progress=35)
            DocumentService._delete_from_milvus(document)
            Chunk.query.filter_by(document_id=document.id).delete()
            db.session.commit()

            vector_ids = DocumentService._store_to_milvus(
                knowledge_base,
                document.id,
                chunks,
                embedding_model=config.embedding_model,
            )
            DocumentIndexService._set_task_status(task, DocumentIndexService.TASK_RUNNING, progress=80)
            DocumentService._save_chunks(
                document,
                knowledge_base.id,
                chunks,
                vector_ids,
                index_task_id=task.id,
            )

            task.result_summary = {
                "chunk_count": len(chunks),
                "vector_count": len(vector_ids),
            }
            task.progress = 100
            task.status = DocumentIndexService.TASK_COMPLETED
            task.finished_at = datetime.now()

            document.status = "completed"
            document.chunk_count = len(chunks)
            document.processed_at = datetime.now()
            document.error_message = None
            document.current_config_id = config.id
            document.current_index_task_id = task.id
            db.session.commit()
            return True, f"索引构建完成，共 {len(chunks)} 个 chunks"
        except Exception as exc:
            db.session.rollback()
            return DocumentIndexService._mark_task_failed(task_id, str(exc), document_id=document.id)

    @staticmethod
    def get_task(task_id: int) -> Optional[DocumentIndexTask]:
        """Fetch task by id."""
        return db.session.get(DocumentIndexTask, task_id)

    @staticmethod
    def get_preview_result(document_id: int) -> Optional[dict]:
        """Return the latest completed preview result for a document."""
        task = (
            DocumentIndexTask.query.filter_by(
                document_id=document_id,
                task_type=DocumentIndexService.PREVIEW_TASK,
                status=DocumentIndexService.TASK_COMPLETED,
            )
            .order_by(DocumentIndexTask.created_at.desc())
            .first()
        )
        return task.result_summary if task else None

    @staticmethod
    def _set_task_status(task: DocumentIndexTask, status: str, progress: int | None = None):
        """Update task running metadata."""
        if task.started_at is None and status == DocumentIndexService.TASK_RUNNING:
            task.started_at = datetime.now()
        task.status = status
        if progress is not None:
            task.progress = progress
        db.session.commit()

    @staticmethod
    def _mark_task_failed(task_id: int, error_message: str, document_id: int | None = None) -> tuple[bool, str]:
        """Persist failure state to task and related document."""
        task = db.session.get(DocumentIndexTask, task_id)
        if task:
            task.status = DocumentIndexService.TASK_FAILED
            task.error_message = error_message
            task.finished_at = datetime.now()
            task.progress = 0

        if document_id is not None:
            document = db.session.get(Document, document_id)
            if document:
                document.status = "failed"
                document.error_message = error_message

        db.session.commit()
        return False, f"任务失败: {error_message}"
