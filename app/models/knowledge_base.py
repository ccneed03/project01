"""
Knowledge Base Model
"""
from datetime import datetime
from app.extensions import db


class KnowledgeBase(db.Model):
    """Knowledge Base model"""
    __tablename__ = "knowledge_bases"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.String(512))
    default_embedding_model = db.Column(db.String(255))

    # Split configuration
    split_separator = db.Column(db.String(256), default="\n\n")  # Text split separator
    chunk_size = db.Column(db.Integer, default=500)
    chunk_overlap = db.Column(db.Integer, default=50)

    # Retrieval configuration
    retrieval_method = db.Column(db.String(32), default="hybrid")  # keyword, vector, hybrid
    rerank_k = db.Column(db.Integer, default=6)  # Top-K for final results
    enable_rerank = db.Column(db.Boolean, default=True)

    # Status
    status = db.Column(db.String(32), default="active")  # active, inactive
    is_deleted = db.Column(db.Boolean, default=False)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    # Foreign keys
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    # Relationships
    creator = db.relationship("User", back_populates="knowledge_bases")
    documents = db.relationship("Document", back_populates="knowledge_base", cascade="all, delete-orphan")
    document_index_configs = db.relationship(
        "DocumentIndexConfig",
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
    )
    document_index_tasks = db.relationship(
        "DocumentIndexTask",
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
    )
    chunks = db.relationship("Chunk", back_populates="knowledge_base", cascade="all, delete-orphan")
    assistant_knowledge_base_configs = db.relationship(
        "AssistantKnowledgeBaseConfig",
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
    )
    chat_sessions = db.relationship("ChatSession", back_populates="knowledge_base")
    search_histories = db.relationship("SearchHistory", back_populates="knowledge_base")
    eval_records = db.relationship("EvalRecord", back_populates="knowledge_base")

    @property
    def document_count(self):
        """Get document count"""
        return Document.query.filter_by(knowledge_base_id=self.id, is_deleted=False).count()

    @property
    def chunk_count(self):
        """Get chunk count"""
        return Chunk.query.filter_by(knowledge_base_id=self.id).count()

    def __repr__(self):
        return f"<KnowledgeBase {self.name}>"


# Import at bottom to avoid circular import
from app.models.document import Document
from app.models.chunk import Chunk
