"""
Assistant application service.
"""
from __future__ import annotations

from typing import Optional

from app.config import Config
from app.extensions import db
from app.models.assistant import Assistant, AssistantKnowledgeBaseConfig
from app.models.knowledge_base import KnowledgeBase


class AssistantService:
    """Manage assistant applications and their retrieval configs."""

    DEFAULT_MODELS = [
        "deepseek-flash",
        "qwen3.5-flash"
    ]

    @staticmethod
    def list_llm_models() -> list[str]:
        """Return available LLM model names."""
        models = []
        configured = (Config.LLM_MODEL or "").strip()
        if configured:
            models.append(configured)
        for name in AssistantService.DEFAULT_MODELS:
            if name not in models:
                models.append(name)
        return models

    @staticmethod
    def create_assistant(
        *,
        name: str,
        description: str = "",
        llm_model: str | None = None,
        system_prompt: str = "",
        opening_message: str = "",
        knowledge_base_id: int | None = None,
        retrieval_mode: str = "hybrid",
        top_k: int = 10,
        enable_rerank: bool = True,
        rerank_model: str | None = None,
        score_threshold: float | None = None,
        created_by: int,
    ) -> tuple[bool, str, Optional[Assistant]]:
        if not name or not name.strip():
            return False, "应用名称不能为空", None

        assistant = Assistant(
            name=name.strip(),
            description=(description or "").strip() or None,
            llm_model=(llm_model or Config.LLM_MODEL).strip(),
            system_prompt=(system_prompt or "").strip() or None,
            opening_message=(opening_message or "").strip() or None,
            created_by=created_by,
            status="published",
        )
        db.session.add(assistant)
        db.session.flush()

        if knowledge_base_id:
            kb = KnowledgeBase.query.filter_by(id=knowledge_base_id, is_deleted=False).first()
            if not kb:
                db.session.rollback()
                return False, "所选知识库不存在", None

            binding = AssistantKnowledgeBaseConfig(
                assistant_id=assistant.id,
                knowledge_base_id=knowledge_base_id,
                retrieval_mode=retrieval_mode or "hybrid",
                top_k=max(int(top_k or 10), 1),
                enable_rerank=bool(enable_rerank),
                rerank_model=(rerank_model or "").strip() or None,
                score_threshold=score_threshold,
            )
            db.session.add(binding)

        db.session.commit()
        return True, "应用创建成功", assistant

    @staticmethod
    def update_assistant(
        assistant_id: int,
        *,
        name: str,
        description: str = "",
        llm_model: str | None = None,
        system_prompt: str = "",
        opening_message: str = "",
        knowledge_base_id: int | None = None,
        retrieval_mode: str = "hybrid",
        top_k: int = 10,
        enable_rerank: bool = True,
        rerank_model: str | None = None,
        score_threshold: float | None = None,
        status: str = "published",
    ) -> tuple[bool, str, Optional[Assistant]]:
        assistant = AssistantService.get_assistant(assistant_id)
        if not assistant:
            return False, "应用不存在", None

        assistant.name = (name or "").strip() or assistant.name
        assistant.description = (description or "").strip() or None
        assistant.llm_model = (llm_model or Config.LLM_MODEL).strip()
        assistant.system_prompt = (system_prompt or "").strip() or None
        assistant.opening_message = (opening_message or "").strip() or None
        assistant.status = status or assistant.status

        existing = assistant.primary_knowledge_base_config
        if knowledge_base_id:
            kb = KnowledgeBase.query.filter_by(id=knowledge_base_id, is_deleted=False).first()
            if not kb:
                db.session.rollback()
                return False, "所选知识库不存在", None
            if existing is None:
                existing = AssistantKnowledgeBaseConfig(assistant_id=assistant.id)
                db.session.add(existing)
            existing.knowledge_base_id = knowledge_base_id
            existing.retrieval_mode = retrieval_mode or "hybrid"
            existing.top_k = max(int(top_k or 10), 1)
            existing.enable_rerank = bool(enable_rerank)
            existing.rerank_model = (rerank_model or "").strip() or None
            existing.score_threshold = score_threshold
        elif existing is not None:
            db.session.delete(existing)

        db.session.commit()
        return True, "应用更新成功", assistant

    @staticmethod
    def delete_assistant(assistant_id: int) -> tuple[bool, str]:
        assistant = AssistantService.get_assistant(assistant_id)
        if not assistant:
            return False, "应用不存在"
        assistant.is_deleted = True
        db.session.commit()
        return True, "应用已删除"

    @staticmethod
    def get_assistant(assistant_id: int) -> Optional[Assistant]:
        return Assistant.query.filter_by(id=assistant_id, is_deleted=False).first()

    @staticmethod
    def list_assistants(
        page: int = 1,
        per_page: int = 20,
        search: str | None = None,
    ) -> tuple[list[Assistant], int]:
        query = Assistant.query.filter_by(is_deleted=False)
        if search:
            query = query.filter(Assistant.name.contains(search))
        total = query.count()
        items = query.order_by(Assistant.updated_at.desc(), Assistant.created_at.desc()).paginate(
            page=page,
            per_page=per_page,
            error_out=False,
        ).items
        return items, total
