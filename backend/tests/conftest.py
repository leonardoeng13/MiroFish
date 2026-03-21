"""
Shared pytest fixtures for the MiroFish test suite
"""

import os
import sys
import types
import tempfile
import pytest

# Ensure the backend package is importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set dummy env vars so Config doesn't fail before tests that don't need real APIs
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("ZEP_API_KEY", "test-zep-key")


def _make_mock_module(name: str, **attrs):
    """Create a lightweight stub module so imports don't fail."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# ---------------------------------------------------------------------------
# Stub out heavy external dependencies that are not installed in the test env.
# This must happen BEFORE any app module is imported.
# ---------------------------------------------------------------------------

def _stub_external_deps():
    # Create a minimal InternalServerError so zep_paging.py can import it
    class _InternalServerError(Exception):
        pass

    zep_mod = _make_mock_module(
        "zep_cloud",
        EpisodeData=object,
        EntityEdgeSourceTarget=object,
        InternalServerError=_InternalServerError,
    )
    zep_client_mod = _make_mock_module("zep_cloud.client", Zep=object)

    sys.modules.setdefault("zep_cloud", zep_mod)
    sys.modules.setdefault("zep_cloud.client", zep_client_mod)
    sys.modules.setdefault("fitz", _make_mock_module("fitz"))


_stub_external_deps()


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after each test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_txt_file(tmp_dir):
    """Create a sample .txt file for parser tests."""
    path = os.path.join(tmp_dir, "sample.txt")
    content = "Hello, world!\nThis is a test file.\nLine three."
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path, content


@pytest.fixture
def sample_md_file(tmp_dir):
    """Create a sample .md file for parser tests."""
    path = os.path.join(tmp_dir, "sample.md")
    content = "# Title\n\nParagraph one.\n\n## Section\n\nParagraph two."
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path, content
