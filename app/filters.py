"""
Template Filters
"""
from datetime import datetime


def register_filters(app):
    """Register Jinja2 template filters"""

    @app.template_filter("datetime")
    def format_datetime(value, format="%Y-%m-%d %H:%M:%S"):
        if value is None:
            return ""
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
            except:
                return value
        return value.strftime(format)

    @app.template_filter("filesize")
    def format_filesize(value):
        if value is None:
            return "0 B"
        for unit in ["B", "KB", "MB", "GB"]:
            if value < 1024:
                return f"{value:.1f} {unit}"
            value /= 1024
        return f"{value:.1f} TB"

    @app.template_filter("truncate_chars")
    def truncate_chars(value, length=100):
        if value is None:
            return ""
        if len(value) <= length:
            return value
        return value[:length] + "..."
