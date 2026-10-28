"""
Document Model
"""
from datetime import datetime
import os
from app.extensions import db


class Document(db.Model):
    """Document model"""
    __tablename__ = "documents"

    id = db.Column(db.Integer, primary_key=True)
    knowledge_base_id = db.Column(db.Integer, db.ForeignKey("knowledge_bases.id"), nullable=False)

    # File info
    title = db.Column(db.String(255))
    filename = db.Column(db.String(256), nullable=False)
    original_filename = db.Column(db.String(256), nullable=False)
    file_path = db.Column(db.String(512), nullable=False)
    file_type = db.Column(db.String(32), nullable=False)  # txt, pdf, docx, md
    file_size = db.Column(db.Integer)  # bytes

    # Processing info
    chunk_count = db.Column(db.Integer, default=0)
    status = db.Column(db.String(32), default="uploaded")
    error_message = db.Column(db.String(512))
    processed_at = db.Column(db.DateTime)
    current_config_id = db.Column(db.Integer)
    current_index_task_id = db.Column(db.Integer)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    is_deleted = db.Column(db.Boolean, default=False)

    # Foreign keys
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    # Celery task ID
    task_id = db.Column(db.String(128))

    # Relationships
    knowledge_base = db.relationship("KnowledgeBase", back_populates="documents")
    creator = db.relationship("User", back_populates="documents")
    chunks = db.relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    index_configs = db.relationship(
        "DocumentIndexConfig",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentIndexConfig.version.desc()",
    )
    index_tasks = db.relationship(
        "DocumentIndexTask",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentIndexTask.created_at.desc()",
    )

    @property
    def status_label(self):
        """Get status label"""
        labels = {
            "pending": "等待处理",
            "parsing": "正在解析",
            "splitting": "正在切分",
            "vectorizing": "正在向量化",
            "uploaded": "已上传",
            "previewing": "预览中",
            "ready_to_index": "待确认索引",
            "indexing": "索引构建中",
            "completed": "已完成",
            "failed": "失败",
        }
        return labels.get(self.status, self.status)

    @property
    def display_title(self) -> str:
        """Prefer custom title, then original filename."""
        return (self.title or self.original_filename or "").strip()

    @property
    def file_size_formatted(self):
        """Get formatted file size"""
        size = self.file_size or 0
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def delete_file(self):
        """Delete the physical file"""
        if self.file_path and os.path.exists(self.file_path):
            os.remove(self.file_path)

    def __repr__(self):
        return f"<Document {self.original_filename}>"


from app.models.chunk import Chunk
