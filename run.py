#!/usr/bin/env python
"""
Flask Application Entry Point
"""
import os
from threading import Thread

from app import create_app
from app.services.warmup_service import warmup_models

app = create_app()


def should_start_warmup(debug_mode: bool) -> bool:
    """Avoid duplicate warmup under Werkzeug's debug reloader."""
    if not debug_mode:
        return True
    return os.environ.get("WERKZEUG_RUN_MAIN") == "true"

if __name__ == "__main__":
    debug_mode = True
    if should_start_warmup(debug_mode):
        Thread(target=warmup_models, daemon=True).start()
    app.run(host="0.0.0.0", port=5001, debug=debug_mode)
