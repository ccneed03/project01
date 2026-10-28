"""
Routes Package
"""
from .auth import auth_bp
from .admin import admin_bp
from .document import document_bp
from .chat import chat_bp
from .search import search_bp
from .api import api_bp
from .agent import agent_bp

__all__ = [
    "auth_bp",
    "admin_bp",
    "document_bp",
    "chat_bp",
    "search_bp",
    "api_bp",
    "agent_bp",
]
