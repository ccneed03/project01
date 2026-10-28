"""
Celery Tasks
"""

from app.extensions import celery
from app.config import Config

celery.conf.update(
    broker_url=Config.CELERY_BROKER_URL,
    result_backend=Config.CELERY_RESULT_BACKEND,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    enable_utc=False,
)


@celery.task(bind=True, max_retries=3)
def preview_document_task(self, task_id: int):
    """Run a preview task for the new document indexing flow."""
    from app import create_app
    app = create_app()

    with app.app_context():
        from app.services.document_index_service import DocumentIndexService

        task = DocumentIndexService.get_task(task_id)
        if task:
            task.celery_task_id = self.request.id
            from app.extensions import db
            db.session.commit()

        try:
            success, message = DocumentIndexService.run_preview_task(task_id)
            if not success:
                raise Exception(message)
            return {"success": True, "message": message}
        except Exception as exc:
            self.retry(exc=exc, countdown=30)


@celery.task(bind=True, max_retries=3)
def index_document_task(self, task_id: int):
    """Run a final indexing task for the new document indexing flow."""
    from app import create_app
    app = create_app()

    with app.app_context():
        from app.services.document_index_service import DocumentIndexService
        from app.extensions import db

        task = DocumentIndexService.get_task(task_id)
        if task:
            task.celery_task_id = self.request.id
            db.session.commit()

        try:
            success, message = DocumentIndexService.run_index_task(task_id)
            if not success:
                raise Exception(message)
            return {"success": True, "message": message}
        except Exception as exc:
            self.retry(exc=exc, countdown=30)
