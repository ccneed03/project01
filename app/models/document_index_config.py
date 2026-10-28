"""
Document index configuration model.
"""
from datetime import datetime

from app.extensions import db


class DocumentIndexConfig(db.Model):
    """Per-document indexing configuration."""

    __tablename__ = "document_index_configs"

    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey("documents.id"), nullable=False)
    knowledge_base_id = db.Column(db.Integer, db.ForeignKey("knowledge_bases.id"), nullable=False)

    embedding_model = db.Column(db.String(255), nullable=False)
    chunk_mode = db.Column(db.String(32), nullable=False, default="paragraph")
    chunk_size = db.Column(db.Integer, nullable=False, default=500)
    chunk_overlap = db.Column(db.Integer, nullable=False, default=50)
    split_separator = db.Column(db.String(255))
    parse_strategy = db.Column(db.String(32), nullable=False, default="auto")

    version = db.Column(db.Integer, nullable=False, default=1)
    is_active = db.Column(db.Boolean, nullable=False, default=True)

    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)

    document = db.relationship("Document", back_populates="index_configs")
    knowledge_base = db.relationship("KnowledgeBase", back_populates="document_index_configs")

    def __repr__(self):
        return f"<DocumentIndexConfig doc={self.document_id} v={self.version}>"
