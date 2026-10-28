"""
Search History Model
"""
from datetime import datetime
from app.extensions import db


class SearchHistory(db.Model):
    """Search History model"""
    __tablename__ = "search_history"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    knowledge_base_id = db.Column(db.Integer, db.ForeignKey("knowledge_bases.id"), nullable=False)

    query = db.Column(db.Text, nullable=False)
    method = db.Column(db.String(32), nullable=False)  # vector, bm25, hybrid
    top_k = db.Column(db.Integer, nullable=False)
    result_count = db.Column(db.Integer, default=0)
    avg_score = db.Column(db.Float)

    # Performance
    latency_ms = db.Column(db.Integer)  # milliseconds

    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False, index=True)

    # Relationships
    user = db.relationship("User", back_populates="search_histories")
    knowledge_base = db.relationship("KnowledgeBase", back_populates="search_histories")

    def __repr__(self):
        return f"<SearchHistory {self.id} - {self.query[:30]}>"
