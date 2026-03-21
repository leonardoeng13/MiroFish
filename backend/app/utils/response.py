"""
API response helpers
====================

Provides a single :func:`error_response` factory that all API routes use to
return consistent error payloads.

Design goals
------------
- Every error response has the same top-level shape:
  ``{"success": False, "error": "<human-readable message>"}``
- Tracebacks are included **only in DEBUG mode** to avoid leaking internal
  implementation details to production clients.
- The function is safe to call outside an application context (e.g. from unit
  tests) — it catches the :exc:`RuntimeError` raised when
  :data:`flask.current_app` is not available.
"""

import traceback
from flask import current_app


def error_response(message: str, status_code: int = 500, exc: Exception = None) -> dict:
    """
    Build a consistent error response dictionary.

    In DEBUG mode the full Python traceback is included so developers can
    diagnose issues quickly.  In production the traceback is omitted to
    avoid leaking internal implementation details.

    Args:
        message: Human-readable error description.
        status_code: HTTP status code (informational; caller must pass to jsonify).
        exc: The exception that triggered this error (optional).

    Returns:
        A dict suitable for passing to ``jsonify()``.
    """
    body: dict = {
        "success": False,
        "error": message,
    }

    # Only include the traceback in DEBUG mode
    debug = False
    try:
        debug = current_app.config.get("DEBUG", False)
    except RuntimeError:
        # Outside application context (e.g., unit tests) — stay safe
        pass

    if debug and exc is not None:
        body["traceback"] = traceback.format_exc()

    return body
