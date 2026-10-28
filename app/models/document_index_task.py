"""
Document index task tracking model.
"""
from datetime import datetime

from app.extensions import db


class DocumentIndexTask(db.Model):
    """Async preview/index task metadata."""

    __tablename__ = "document_index_tasks"

    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey("documents.id"), nullable=False)
    knowledge_base_id = db.Column(db.Integer, db.ForeignKey("knowledge_bases.id"), nullable=False)
    config_id = db.Column(db.Integer, db.ForeignKey("document_index_configs.id"), nullable=False)

    task_type = db.Column(db.String(32), nullable=False)  # preview, index
    status = db.Column(db.String(32), nullable=False, default="pending")
    celery_task_id = db.Column(db.String(128))
    progress = db.Column(db.Integer, nullable=False, default=0)
    error_message = db.Column(db.String(1024))
    result_summary = db.Column(db.JSON)

    started_at = db.Column(db.DateTime)
    finished_at = db.Column(db.DateTime)
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)

    document = db.relationship("Document", back_populates="index_tasks")
    knowledge_base = db.relationship("KnowledgeBase", back_populates="document_index_tasks")
    config = db.relationship("DocumentIndexConfig")
    chunks = db.relationship("Chunk", back_populates="index_task")

    def __repr__(self):
        return f"<DocumentIndexTask {self.task_type} doc={self.document_id} status={self.status}>"
