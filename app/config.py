"""
Flask Configuration
"""
import os
from datetime import timedelta
from dotenv import load_dotenv


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))


class Config:
    """Base configuration"""
    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "rag-app-secret-key-change-in-production")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///app.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Session
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"

    # File Upload
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = {"txt", "pdf", "docx", "md"}

    # Milvus (Global Configuration)
    MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
    MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:Milvus")
    MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_collection")

    # Ollama Embedding Model
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    # qwen3-embedding:0.6b returns 1024-dimensional vectors.
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

    # Rerank Model (Global Configuration)
    RERANK_MODEL_PATH = "D:/models/bge-reranker-v2-m3"

    # Celery
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

    # LLM
    LLM_API_KEY = os.getenv("D_API_KEY", "")
    LLM_BASE_URL = os.getenv("D_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    LLM_MODEL = os.getenv("D_MODEL", "qwen3.5-flash")

    # Agent (demo scope, intentionally small)
    AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "5"))
    AGENT_ACTION_TTL_MINUTES = int(os.getenv("AGENT_ACTION_TTL_MINUTES", "15"))
    AGENT_HISTORY_LIMIT = int(os.getenv("AGENT_HISTORY_LIMIT", "6"))
    AGENT_RETURN_WINDOW_DAYS = int(os.getenv("AGENT_RETURN_WINDOW_DAYS", "7"))


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SESSION_COOKIE_SECURE = True


config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
