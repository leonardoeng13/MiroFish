"""
Integration tests for the Flask API endpoints using the test client.

These tests do NOT require real LLM / Zep connectivity.
They cover routing, request validation, and error-response shapes.
"""

import os
import json
import io
import pytest

# Env vars must be set before importing app modules
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("ZEP_API_KEY", "test-zep-key")

from app import create_app


@pytest.fixture(scope="module")
def app():
    """Create a Flask test application."""
    application = create_app()
    application.config["TESTING"] = True
    application.config["DEBUG"] = False
    return application


@pytest.fixture(scope="module")
def client(app):
    """Flask test client."""
    return app.test_client()


# ===========================================================================
# /health
# ===========================================================================

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_json(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert data is not None

    def test_health_contains_status(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert "status" in data

    def test_health_contains_service_name(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert "service" in data

    def test_health_status_value(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert data["status"] == "ok"

    def test_health_details_returns_200(self, client):
        resp = client.get("/health/details")
        assert resp.status_code == 200

    def test_health_details_contains_config_status(self, client):
        resp = client.get("/health/details")
        data = resp.get_json()
        assert "config" in data

    def test_health_details_contains_version(self, client):
        resp = client.get("/health/details")
        data = resp.get_json()
        assert "version" in data


# ===========================================================================
# /api/graph – project management
# ===========================================================================

class TestGraphProjectEndpoints:
    def test_list_projects_returns_200(self, client):
        resp = client.get("/api/graph/project/list")
        assert resp.status_code == 200

    def test_list_projects_has_success_flag(self, client):
        resp = client.get("/api/graph/project/list")
        data = resp.get_json()
        assert data.get("success") is True

    def test_list_projects_has_data_array(self, client):
        resp = client.get("/api/graph/project/list")
        data = resp.get_json()
        assert isinstance(data.get("data"), list)

    def test_get_nonexistent_project_returns_404(self, client):
        resp = client.get("/api/graph/project/proj_doesnotexist")
        assert resp.status_code == 404

    def test_get_nonexistent_project_has_error_field(self, client):
        resp = client.get("/api/graph/project/proj_doesnotexist")
        data = resp.get_json()
        assert data.get("success") is False
        assert "error" in data

    def test_delete_nonexistent_project_returns_404(self, client):
        resp = client.delete("/api/graph/project/proj_doesnotexist")
        assert resp.status_code == 404

    def test_reset_nonexistent_project_returns_404(self, client):
        resp = client.post("/api/graph/project/proj_doesnotexist/reset")
        assert resp.status_code == 404

    def test_ontology_generate_without_files_returns_400(self, client):
        resp = client.post(
            "/api/graph/ontology/generate",
            data={"simulation_requirement": "Predict outcome"},
        )
        assert resp.status_code == 400

    def test_ontology_generate_without_requirement_returns_400(self, client):
        data = {
            "simulation_requirement": "",
        }
        file_data = io.BytesIO(b"sample text content")
        resp = client.post(
            "/api/graph/ontology/generate",
            data={**data, "files": (file_data, "test.txt")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert body.get("success") is False

    def test_ontology_generate_unsupported_file_returns_400(self, client):
        """Uploading only unsupported file types should be rejected."""
        file_data = io.BytesIO(b"<html>content</html>")
        resp = client.post(
            "/api/graph/ontology/generate",
            data={
                "simulation_requirement": "Test requirement",
                "files": (file_data, "page.html"),
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400


# ===========================================================================
# /api/simulation – validation
# ===========================================================================

class TestSimulationEndpoints:
    def test_create_simulation_without_project_id_returns_400(self, client):
        resp = client.post(
            "/api/simulation/create",
            json={},
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data.get("success") is False

    def test_create_simulation_nonexistent_project_returns_404(self, client):
        resp = client.post(
            "/api/simulation/create",
            json={"project_id": "proj_doesnotexist"},
            content_type="application/json",
        )
        assert resp.status_code == 404

    def test_get_entities_nonexistent_graph_returns_500_or_error(self, client):
        """Requesting entities for a non-existent graph should return an error (not a 200)."""
        resp = client.get("/api/simulation/entities/nonexistent_graph_id")
        # Should be either 500 (Zep error) or 400/404
        data = resp.get_json()
        assert data.get("success") is False


# ===========================================================================
# /api/report – validation
# ===========================================================================

class TestReportEndpoints:
    def test_generate_without_simulation_id_returns_400(self, client):
        resp = client.post(
            "/api/report/generate",
            json={},
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data.get("success") is False

    def test_generate_nonexistent_simulation_returns_404(self, client):
        resp = client.post(
            "/api/report/generate",
            json={"simulation_id": "sim_doesnotexist"},
            content_type="application/json",
        )
        assert resp.status_code == 404

    def test_get_report_nonexistent_simulation_returns_404_or_error(self, client):
        resp = client.get("/api/report/by-simulation/sim_doesnotexist")
        data = resp.get_json()
        assert data.get("success") is False
