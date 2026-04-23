"""
Configuration Management
========================

Loads runtime configuration from a ``.env`` file located at the project
root (``MiroFish/.env``).  If no file is found the module falls back to
OS-level environment variables, which is the expected behaviour in
container/production deployments.

Priority order
--------------
1. ``MiroFish/.env``  (relative to ``backend/app/config.py``)
2. Any ``.env`` discoverable by ``python-dotenv`` (current working directory)
3. OS environment variables (always present)

All configuration values are exposed as class attributes on :class:`Config`
so that they can be injected into the Flask app via
``app.config.from_object(Config)``.
"""

import os
from dotenv import load_dotenv

# Load project root .env file
# Path: MiroFish/.env (relative to backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # If no .env in root directory, try loading environment variables (for production)
    load_dotenv(override=True)


class Config:
    """Flask configuration class.

    All settings are read from environment variables (loaded above from
    ``.env``).  Defaults are provided for non-secret values so the server
    can start in development mode without a fully populated ``.env`` file.

    Required secrets (validated by :meth:`validate`):
        - ``LLM_API_KEY``  — API key for the OpenAI-compatible LLM provider
        - ``ZEP_API_KEY``  — API key for the Zep knowledge-graph service
    """
    
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # JSON configuration - disable ASCII escaping so non-ASCII chars display directly (not as \uXXXX)
    JSON_AS_ASCII = False
    
    # LLM configuration (unified OpenAI format)
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')
    
    # Zep configuration
    ZEP_API_KEY = os.environ.get('ZEP_API_KEY')
    # Optional: URL for a self-hosted Zep CE instance (e.g. http://localhost:8000).
    # When set, the Zep client connects to this local server instead of Zep Cloud.
    # The ZEP_API_KEY is still required by the SDK but can be any non-empty string
    # when the local server has authentication disabled.
    ZEP_BASE_URL = os.environ.get('ZEP_BASE_URL')
    
    # File upload configuration
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}
    
    # Text processing configuration
    DEFAULT_CHUNK_SIZE = 500  # Default chunk size
    DEFAULT_CHUNK_OVERLAP = 50  # Default overlap size
    
    # OASIS simulation configuration
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')
    
    # OASIS platform available actions configuration
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]
    
    # Report Agent configuration
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))
    
    @classmethod
    def validate(cls):
        """Validate required configuration.

        Returns:
            list[str]: A (possibly empty) list of human-readable error messages.
                An empty list means the configuration is valid and the server
                can make real LLM/Zep API calls.  The ``/health/details``
                endpoint exposes these errors without leaking secret values.
        """
        errors = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY is not configured")
        if not cls.ZEP_API_KEY:
            if cls.ZEP_BASE_URL:
                errors.append(
                    "ZEP_API_KEY is not configured. "
                    "When using a local Zep instance (ZEP_BASE_URL is set), "
                    "you may set ZEP_API_KEY to any non-empty string if authentication is disabled."
                )
            else:
                errors.append("ZEP_API_KEY is not configured")
        return errors
