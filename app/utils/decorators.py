"""
Utility Decorators
"""
from functools import wraps
from flask import abort, flash, redirect, url_for
from flask_login import current_user


def permission_required(resource: str, action: str):
    """Decorator to check if user has permission"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                flash("请先登录", "warning")
                return redirect(url_for("auth.login"))

            if not current_user.has_permission(resource, action):
                flash("您没有权限执行此操作", "danger")
                abort(403)

            return f(*args, **kwargs)
        return decorated_function
    return decorator


def admin_required(f):
    """Decorator to check if user is admin or superuser"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash("请先登录", "warning")
            return redirect(url_for("auth.login"))

        if not (current_user.is_superuser or any(r.name in ["admin", "superadmin"] for r in current_user.roles)):
            flash("您没有权限执行此操作", "danger")
            abort(403)

        return f(*args, **kwargs)
    return decorated_function


def superuser_required(f):
    """Decorator to check if user is superuser"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash("请先登录", "warning")
            return redirect(url_for("auth.login"))

        if not current_user.is_superuser:
            flash("您没有权限执行此操作", "danger")
            abort(403)

        return f(*args, **kwargs)
    return decorated_function


def ajax_only(f):
    """Decorator to restrict view to AJAX requests"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from flask import request
        if not request.is_xhr:
            abort(400)
        return f(*args, **kwargs)
    return decorated_function
