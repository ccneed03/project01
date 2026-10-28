"""
Chat Models
"""
from datetime import datetime
from app.extensions import db


class ChatSession(db.Model):
    """Chat Session model"""
    __tablename__ = "chat_sessions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    knowledge_base_id = db.Column(db.Integer, db.ForeignKey("knowledge_bases.id"))
    assistant_id = db.Column(db.Integer, db.ForeignKey("assistants.id"))

    title = db.Column(db.String(256), default="新对话")
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    user = db.relationship("User", back_populates="chat_sessions")
    knowledge_base = db.relationship("KnowledgeBase", back_populates="chat_sessions")
    assistant = db.relationship("Assistant", back_populates="chat_sessions")
    messages = db.relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at")

    @property
    def message_count(self):
        """Get message count"""
        return len(self.messages)

    def __repr__(self):
        return f"<ChatSession {self.id} - {self.title}>"


class ChatMessage(db.Model):
    """Chat Message model"""
    __tablename__ = "chat_messages"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("chat_sessions.id"), nullable=False)

    role = db.Column(db.String(32), nullable=False)  # user, assistant
    content = db.Column(db.Text, nullable=False)
    reasoning_content = db.Column(db.Text)

    # References (JSON): [{"chunk_id": 1, "document": "xxx.pdf", "score": 0.92}]
    references = db.Column(db.JSON)

    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)

    # Relationships
    session = db.relationship("ChatSession", back_populates="messages")

    def __repr__(self):
        return f"<ChatMessage {self.id} - {self.role}>"
