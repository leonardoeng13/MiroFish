"""
Utilities module
================

General-purpose helpers used throughout the MiroFish backend.

Exported symbols
----------------
- :class:`~app.utils.file_parser.FileParser`
    Extract plain text from PDF, Markdown, and TXT files.
- :class:`~app.utils.llm_client.LLMClient`
    Thin wrapper around the OpenAI-compatible chat API.
- :func:`~app.utils.response.error_response`
    Build a consistent ``{"success": False, "error": …}`` dict;
    includes a full traceback in DEBUG mode only.
- :func:`~app.utils.validators.parse_request`
    Validate a JSON request body against a Pydantic model.
- :class:`~app.utils.prediction_evaluator.PredictionEvidenceTracker`
    Parse ``agent_log.jsonl`` to compute how evidence-backed a report is.
- :class:`~app.utils.prediction_evaluator.PredictionEvidence`
    Data-class carrying the evidence metrics returned by the tracker.
"""

from .file_parser import FileParser
from .llm_client import LLMClient
from .response import error_response
from .validators import parse_request
from .prediction_evaluator import PredictionEvidenceTracker, PredictionEvidence

__all__ = [
    'FileParser',
    'LLMClient',
    'error_response',
    'parse_request',
    'PredictionEvidenceTracker',
    'PredictionEvidence',
]
