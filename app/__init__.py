"""
Flask Application Factory
"""
from flask import Flask
from sqlalchemy import inspect, text
from app.extensions import db, login_manager, csrf, celery
from app.config import Config


def _ensure_schema_updates():
    """Apply lightweight schema updates for existing local databases."""
    inspector = inspect(db.engine)
    table_names = set(inspector.get_table_names())

    def ensure_column(table_name: str, column_name: str, ddl: str):
        if table_name not in table_names:
            return
        columns = {column["name"] for column in inspector.get_columns(table_name)}
        if column_name not in columns:
            db.session.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {ddl}"))
            db.session.commit()

    ensure_column("chat_messages", "reasoning_content", "reasoning_content TEXT")
    ensure_column("knowledge_bases", "default_embedding_model", "default_embedding_model VARCHAR(255)")
    ensure_column("documents", "title", "title VARCHAR(255)")
    ensure_column("documents", "current_config_id", "current_config_id INTEGER")
    ensure_column("documents", "current_index_task_id", "current_index_task_id INTEGER")
    ensure_column("chunks", "index_task_id", "index_task_id INTEGER")
    ensure_column("chat_sessions", "assistant_id", "assistant_id INTEGER")


def create_app(config_class=Config):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)

    # Configure login
    login_manager.login_view = "auth.login"
    login_manager.login_message = "请先登录"
    login_manager.login_message_category = "warning"

    @login_manager.user_loader
    def load_user(user_id):
        from app.models.user import User
        return db.session.get(User, int(user_id))

    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.admin import admin_bp
    from app.routes.document import document_bp
    from app.routes.chat import chat_bp
    from app.routes.search import search_bp
    from app.routes.api import api_bp
    from app.routes.agent import agent_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(document_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(agent_bp, url_prefix="/api")

    # Register template filters
    from app.filters import register_filters
    register_filters(app)

    # Create tables and init data
    with app.app_context():
        from app import models as _models  # noqa: F401 - ensure all model metadata is registered
        db.create_all()
        _ensure_schema_updates()
        from app.services.auth_service import AuthService
        AuthService.init_default_data()

    return app
