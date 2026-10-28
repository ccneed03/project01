"""
User Model
"""
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from app.extensions import db, login_manager


class User(UserMixin, db.Model):
    """User model"""
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(128), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    is_superuser = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    roles = db.relationship("Role", secondary="user_roles", back_populates="users")
    knowledge_bases = db.relationship("KnowledgeBase", back_populates="creator")
    assistants = db.relationship("Assistant", back_populates="creator")
    documents = db.relationship("Document", back_populates="creator")
    chat_sessions = db.relationship("ChatSession", back_populates="user")
    search_histories = db.relationship("SearchHistory", back_populates="user")
    eval_records = db.relationship("EvalRecord", back_populates="user")

    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Check password"""
        return check_password_hash(self.password_hash, password)

    def has_permission(self, resource: str, action: str) -> bool:
        """Check if user has permission"""
        if self.is_superuser:
            return True

        for role in self.roles:
            for perm in role.permissions:
                if perm.resource == resource and perm.action == action:
                    return True
        return False

    def __repr__(self):
        return f"<User {self.username}>"


@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login"""
    return db.session.get(User, int(user_id))
