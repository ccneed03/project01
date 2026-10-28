"""
Flask Extensions
"""
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect
from celery import Celery

db = SQLAlchemy()
login_manager = LoginManager()
csrf = CSRFProtect()
celery = Celery(__name__)
