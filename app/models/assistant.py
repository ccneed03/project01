"""
Assistant application models.
"""
from datetime import datetime

from app.config import Config
from app.extensions import db


class Assistant(db.Model):
    """AI assistant / application."""

    __tablename__ = "assistants"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.String(512))
    llm_model = db.Column(db.String(128), nullable=False, default=lambda: Config.LLM_MODEL)
    system_prompt = db.Column(db.Text)
    opening_message = db.Column(db.Text)
    status = db.Column(db.String(32), default="published", nullable=False)
    is_deleted = db.Column(db.Boolean, default=False, nullable=False)
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    creator = db.relationship("User", back_populates="assistants")
    knowledge_base_configs = db.relationship(
        "AssistantKnowledgeBaseConfig",
        back_populates="assistant",
        cascade="all, delete-orphan",
        order_by="AssistantKnowledgeBaseConfig.id",
    )
    chat_sessions = db.relationship("ChatSession", back_populates="assistant")

    @property
    def primary_knowledge_base_config(self):
        """Return the first configured knowledge base binding."""
        return self.knowledge_base_configs[0] if self.knowledge_base_configs else None

    @property
    def knowledge_base(self):
        """Shortcut to bound knowledge base."""
        config = self.primary_knowledge_base_config
        return config.knowledge_base if config else None

    def __repr__(self):
        return f"<Assistant {self.name}>"


class AssistantKnowledgeBaseConfig(db.Model):
    """Retrieval config used when an assistant mounts a knowledge base."""

    __tablename__ = "assistant_knowledge_base_configs"

    id = db.Column(db.Integer, primary_key=True)
    assistant_id = db.Column(db.Integer, db.ForeignKey("assistants.id"), nullable=False)
    knowledge_base_id = db.Column(db.Integer, db.ForeignKey("knowledge_bases.id"), nullable=False)
    retrieval_mode = db.Column(db.String(32), default="hybrid", nullable=False)
    top_k = db.Column(db.Integer, default=10, nullable=False)
    enable_rerank = db.Column(db.Boolean, default=True, nullable=False)
    rerank_model = db.Column(db.String(255))
    score_threshold = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    assistant = db.relationship("Assistant", back_populates="knowledge_base_configs")
    knowledge_base = db.relationship("KnowledgeBase", back_populates="assistant_knowledge_base_configs")

    def __repr__(self):
        return f"<AssistantKnowledgeBaseConfig assistant={self.assistant_id} kb={self.knowledge_base_id}>"
