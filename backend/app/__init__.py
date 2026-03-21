"""
MiroFish Backend — Flask Application Factory
=============================================

This module is the entry-point for the Flask server.  It follows the
*application factory* pattern so that multiple app instances can be created
in tests or worker processes without sharing global state.

Startup sequence
----------------
1. Suppress third-party warnings before any heavy imports.
2. Create the Flask app from a Config object.
3. Configure JSON encoding (Unicode passthrough).
4. Set up rotating-file + console logging via :func:`setup_logger`.
5. Enable CORS for all ``/api/*`` routes (development-friendly ``"*"``).
6. Register the simulation process cleanup hook (``atexit``).
7. Attach request/response logging middleware.
8. Register the three API blueprints under ``/api/graph``,
   ``/api/simulation``, and ``/api/report``.
9. Expose ``/health`` and ``/health/details`` endpoints.
"""

import os
import warnings

# Suppress multiprocessing resource_tracker warnings (from third-party libraries like transformers)
# Must be set before all other imports
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from flask import Flask, request
from flask_cors import CORS

from .config import Config
from .utils.logger import setup_logger, get_logger


def create_app(config_class=Config):
    """Flask application factory.

    Creates and fully configures a :class:`~flask.Flask` application instance.
    Using a factory instead of a module-level ``app = Flask(__name__)`` allows:

    - Multiple isolated instances in tests (each test can get a fresh app).
    - Different configurations per environment without changing source code.
    - Deferred imports that avoid circular-dependency issues.

    Args:
        config_class: A configuration class (or object) compatible with
            :meth:`~flask.Flask.config.from_object`.  Defaults to
            :class:`~app.config.Config` which reads from environment variables.

    Returns:
        A fully initialised :class:`~flask.Flask` application instance ready
        to be handed to a WSGI server (``gunicorn``, ``waitress``) or the
        Werkzeug development server.
    """
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Set JSON encoding: ensure non-ASCII characters are displayed directly (not as \uXXXX format)
    # Flask >= 2.3 uses app.json.ensure_ascii, older versions use JSON_AS_ASCII config
    if hasattr(app, 'json') and hasattr(app.json, 'ensure_ascii'):
        app.json.ensure_ascii = False
    
    # Set up logging
    logger = setup_logger('mirofish')
    
    # Only print startup info in reloader subprocess (avoid printing twice in debug mode)
    is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    debug_mode = app.config.get('DEBUG', False)
    should_log_startup = not debug_mode or is_reloader_process
    
    if should_log_startup:
        logger.info("=" * 50)
        logger.info("MiroFish Backend starting...")
        logger.info("=" * 50)
    
    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Register simulation process cleanup (ensure all simulation processes are terminated on server shutdown)
    from .services.simulation_runner import SimulationRunner
    SimulationRunner.register_cleanup()
    if should_log_startup:
        logger.info("Simulation process cleanup function registered")
    
    # Request logging middleware
    @app.before_request
    def log_request():
        logger = get_logger('mirofish.request')
        logger.debug(f"Request: {request.method} {request.path}")
        if request.content_type and 'json' in request.content_type:
            logger.debug(f"Request body: {request.get_json(silent=True)}")
    
    @app.after_request
    def log_response(response):
        logger = get_logger('mirofish.request')
        logger.debug(f"Response: {response.status_code}")
        return response
    
    # Register blueprints
    from .api import graph_bp, simulation_bp, report_bp
    app.register_blueprint(graph_bp, url_prefix='/api/graph')
    app.register_blueprint(simulation_bp, url_prefix='/api/simulation')
    app.register_blueprint(report_bp, url_prefix='/api/report')
    
    # Health check
    @app.route('/health')
    def health():
        from datetime import datetime, timezone
        return {
            'status': 'ok',
            'service': 'MiroFish Backend',
            'version': '0.1.0',
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }

    @app.route('/health/details')
    def health_details():
        """
        Detailed health check.

        Returns configuration status and whether required credentials are
        present, without exposing the actual secret values.
        """
        from datetime import datetime, timezone
        from .config import Config

        config_errors = Config.validate()
        config_ok = len(config_errors) == 0

        return {
            'status': 'ok' if config_ok else 'degraded',
            'service': 'MiroFish Backend',
            'version': '0.1.0',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'config': {
                'llm_api_key_set': bool(Config.LLM_API_KEY),
                'zep_api_key_set': bool(Config.ZEP_API_KEY),
                'llm_base_url': Config.LLM_BASE_URL,
                'llm_model': Config.LLM_MODEL_NAME,
                'errors': config_errors,
            },
        }
    
    if should_log_startup:
        logger.info("MiroFish Backend startup complete")
    
    return app

