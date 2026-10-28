"""
Evaluation Record Model
"""
from datetime import datetime
from app.extensions import db


class EvalRecord(db.Model):
    """Evaluation Record model"""
    __tablename__ = "eval_records"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    knowledge_base_id = db.Column(db.Integer, db.ForeignKey("knowledge_bases.id"), nullable=False)

    query = db.Column(db.Text, nullable=False)
    reference_answer = db.Column(db.Text)
    rag_answer = db.Column(db.Text)

    # Evaluation metrics (from Ragas)
    faithfulness = db.Column(db.Float)
    answer_correctness = db.Column(db.Float)
    context_precision = db.Column(db.Float)
    context_recall = db.Column(db.Float)
    answer_relevancy = db.Column(db.Float)

    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False, index=True)

    # Relationships
    user = db.relationship("User", back_populates="eval_records")
    knowledge_base = db.relationship("KnowledgeBase", back_populates="eval_records")

    def __repr__(self):
        return f"<EvalRecord {self.id} - Faithfulness: {self.faithfulness}>"
