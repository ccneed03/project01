"""
Chunk Model
"""
from datetime import datetime
from app.extensions import db


class Chunk(db.Model):
    """Chunk model"""
    __tablename__ = "chunks"

    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey("documents.id"), nullable=False)
    knowledge_base_id = db.Column(db.Integer, db.ForeignKey("knowledge_bases.id"), nullable=False)
    index_task_id = db.Column(db.Integer, db.ForeignKey("document_index_tasks.id"))

    # Content
    content = db.Column(db.Text, nullable=False)
    vector_id = db.Column(db.String(128))  # Vector ID in Milvus

    # Metadata
    chunk_index = db.Column(db.Integer, nullable=False)  # Order in document
    chunk_metadata = db.Column(db.JSON)  # Additional metadata (page number, etc.)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)

    # Relationships
    document = db.relationship("Document", back_populates="chunks")
    knowledge_base = db.relationship("KnowledgeBase", back_populates="chunks")
    index_task = db.relationship("DocumentIndexTask", back_populates="chunks")

    @property
    def content_preview(self, length=100):
        """Get content preview"""
        if len(self.content) <= length:
            return self.content
        return self.content[:length] + "..."

    @property
    def metadata_dict(self) -> dict:
        """Return chunk metadata as a mutable dict."""
        return dict(self.chunk_metadata or {})

    @property
    def tags(self) -> list[str]:
        """Return normalized chunk tags."""
        tags = self.metadata_dict.get("tags", [])
        if isinstance(tags, list):
            return [str(tag).strip() for tag in tags if str(tag).strip()]
        return []

    @property
    def note(self) -> str:
        """Return editor note stored in metadata."""
        return str(self.metadata_dict.get("note", "")).strip()

    def __repr__(self):
        return f"<Chunk {self.id} - Doc {self.document_id}>"
