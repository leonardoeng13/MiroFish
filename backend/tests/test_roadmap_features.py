"""
Tests for the 4 roadmap features added in the evolution sprint:

1. POST /api/report/<report_id>/retry          – retry failed report
2. GET  /api/report/<report_id>/export/html    – HTML export
3. DELETE /api/simulation/<simulation_id>      – simulation delete
4. GET  /api/report/compare                    – report comparison

Tests use the existing Flask test-client fixture from conftest.py.
"""

import json
import os
import pytest

from app.services.report_agent import ReportManager, Report, ReportStatus, ReportOutline, ReportSection
from app.services.simulation_manager import SimulationManager, SimulationState, SimulationStatus


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_report(
    tmp_path,
    monkeypatch,
    report_id: str,
    status: ReportStatus = ReportStatus.COMPLETED,
    with_evidence: bool = True,
    sections: list | None = None,
) -> Report:
    """Save a minimal Report to a temp directory and return it."""
    monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(tmp_path / "reports"))

    outline_sections = sections or [
        ReportSection(title="Section Alpha", content="Alpha content."),
        ReportSection(title="Section Beta", content="Beta content."),
    ]
    outline = ReportOutline(
        title="Test Report Title",
        summary="Summary sentence.",
        sections=outline_sections,
    )
    ev = {
        "report_id": report_id,
        "total_tool_calls": 7,
        "unique_tools_used": ["insight_forge", "panorama_search", "quick_search", "interview_agents"],
        "facts_retrieved": 15,
        "agents_interviewed": 3,
        "evidence_score": 82.0,
        "is_evidence_based": True,
        "sections": [],
    } if with_evidence else None

    report = Report(
        report_id=report_id,
        simulation_id="sim_test",
        graph_id="graph_test",
        simulation_requirement="test scenario",
        status=status,
        outline=outline,
        markdown_content="# Test Report Title\n\n> Summary sentence.\n\n## Section Alpha\n\nAlpha content.\n\n## Section Beta\n\nBeta content.\n",
        created_at="2026-01-01T00:00:00",
        completed_at="2026-01-01T01:00:00",
        evidence_summary=ev,
    )
    ReportManager.save_report(report)
    return report


def _make_simulation(tmp_path, monkeypatch, simulation_id: str, status: SimulationStatus = SimulationStatus.READY) -> SimulationState:
    """Save a minimal SimulationState and return it."""
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(tmp_path / "simulations"))

    state = SimulationState(
        simulation_id=simulation_id,
        project_id="proj_test",
        graph_id="graph_test",
        status=status,
    )
    manager = SimulationManager()
    manager._save_simulation_state(state)
    return state


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    import os as _os
    _os.environ.setdefault("LLM_API_KEY", "test-key")
    _os.environ.setdefault("ZEP_API_KEY", "test-zep-key")
    from app import create_app
    application = create_app()
    application.config["TESTING"] = True
    return application


@pytest.fixture(scope="module")
def client(app):
    return app.test_client()


# ──────────────────────────────────────────────────────────────────────────────
# Feature 3: SimulationManager.delete_simulation
# ──────────────────────────────────────────────────────────────────────────────

class TestSimulationManagerDelete:
    def test_delete_existing_simulation_returns_true(self, tmp_path, monkeypatch):
        _make_simulation(tmp_path, monkeypatch, "sim_del_001")
        manager = SimulationManager()
        result = manager.delete_simulation("sim_del_001")
        assert result is True

    def test_deleted_simulation_no_longer_found(self, tmp_path, monkeypatch):
        _make_simulation(tmp_path, monkeypatch, "sim_del_002")
        manager = SimulationManager()
        manager.delete_simulation("sim_del_002")
        assert manager.get_simulation("sim_del_002") is None

    def test_delete_nonexistent_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(tmp_path / "simulations_empty"))
        manager = SimulationManager()
        result = manager.delete_simulation("sim_doesnotexist")
        assert result is False

    def test_delete_removes_directory(self, tmp_path, monkeypatch):
        _make_simulation(tmp_path, monkeypatch, "sim_del_003")
        manager = SimulationManager()
        sim_dir = manager._get_simulation_dir("sim_del_003")
        assert os.path.exists(sim_dir)
        manager.delete_simulation("sim_del_003")
        assert not os.path.exists(sim_dir)

    def test_delete_purges_in_memory_cache(self, tmp_path, monkeypatch):
        _make_simulation(tmp_path, monkeypatch, "sim_del_004")
        manager = SimulationManager()
        # Force load into cache
        _ = manager.get_simulation("sim_del_004")
        assert "sim_del_004" in manager._simulations
        manager.delete_simulation("sim_del_004")
        assert "sim_del_004" not in manager._simulations


# ──────────────────────────────────────────────────────────────────────────────
# Feature 3: DELETE /api/simulation/<simulation_id> endpoint
# ──────────────────────────────────────────────────────────────────────────────

class TestSimulationDeleteEndpoint:
    def test_delete_nonexistent_returns_404(self, client):
        resp = client.delete("/api/simulation/sim_doesnotexist_xyz")
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["success"] is False

    def test_delete_running_simulation_returns_409(self, client, tmp_path, monkeypatch):
        _make_simulation(tmp_path, monkeypatch, "sim_running_001", status=SimulationStatus.RUNNING)
        resp = client.delete("/api/simulation/sim_running_001")
        assert resp.status_code == 409
        data = resp.get_json()
        assert data["success"] is False
        assert "running" in data["error"].lower()

    def test_delete_ready_simulation_returns_200(self, client, tmp_path, monkeypatch):
        _make_simulation(tmp_path, monkeypatch, "sim_ready_001", status=SimulationStatus.READY)
        resp = client.delete("/api/simulation/sim_ready_001")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True


# ──────────────────────────────────────────────────────────────────────────────
# Feature 2: ReportManager.render_html
# ──────────────────────────────────────────────────────────────────────────────

class TestRenderHtml:
    def test_returns_html_string(self, tmp_path, monkeypatch):
        report = _make_report(tmp_path, monkeypatch, "report_html_001")
        html = ReportManager.render_html(report)
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html

    def test_contains_report_title(self, tmp_path, monkeypatch):
        report = _make_report(tmp_path, monkeypatch, "report_html_002")
        html = ReportManager.render_html(report)
        assert "Test Report Title" in html

    def test_contains_section_headings(self, tmp_path, monkeypatch):
        report = _make_report(tmp_path, monkeypatch, "report_html_003")
        html = ReportManager.render_html(report)
        assert "Section Alpha" in html
        assert "Section Beta" in html

    def test_evidence_badge_present_when_evidence_exists(self, tmp_path, monkeypatch):
        report = _make_report(tmp_path, monkeypatch, "report_html_004", with_evidence=True)
        html = ReportManager.render_html(report)
        assert '<div class="evidence-badge">' in html
        assert "82.0" in html

    def test_no_evidence_badge_when_no_evidence(self, tmp_path, monkeypatch):
        report = _make_report(tmp_path, monkeypatch, "report_html_005", with_evidence=False)
        html = ReportManager.render_html(report)
        # The CSS class name still appears in the stylesheet, but the badge DIV must not
        assert '<div class="evidence-badge">' not in html

    def test_blockquote_rendered(self, tmp_path, monkeypatch):
        report = _make_report(tmp_path, monkeypatch, "report_html_006")
        html = ReportManager.render_html(report)
        assert "<blockquote>" in html

    def test_empty_markdown_still_valid_html(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(tmp_path / "reports_empty"))
        report = Report(
            report_id="report_html_007",
            simulation_id="sim_x",
            graph_id="g_x",
            simulation_requirement="",
            status=ReportStatus.COMPLETED,
            markdown_content="",
        )
        html = ReportManager.render_html(report)
        assert "<!DOCTYPE html>" in html
        assert "<body>" in html


# ──────────────────────────────────────────────────────────────────────────────
# Feature 2: GET /api/report/<report_id>/export/html endpoint
# ──────────────────────────────────────────────────────────────────────────────

class TestExportHtmlEndpoint:
    def test_export_nonexistent_returns_404(self, client):
        resp = client.get("/api/report/report_noexist_xyz/export/html")
        assert resp.status_code == 404

    def test_export_incomplete_report_returns_400(self, client, tmp_path, monkeypatch):
        _make_report(tmp_path, monkeypatch, "report_exp_001", status=ReportStatus.GENERATING)
        resp = client.get("/api/report/report_exp_001/export/html")
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_export_completed_report_returns_html(self, client, tmp_path, monkeypatch):
        _make_report(tmp_path, monkeypatch, "report_exp_002", status=ReportStatus.COMPLETED)
        resp = client.get("/api/report/report_exp_002/export/html")
        assert resp.status_code == 200
        assert b"<!DOCTYPE html>" in resp.data
        assert b"Test Report Title" in resp.data


# ──────────────────────────────────────────────────────────────────────────────
# Feature 4: GET /api/report/compare endpoint
# ──────────────────────────────────────────────────────────────────────────────

class TestCompareReportsEndpoint:
    def test_missing_params_returns_400(self, client):
        resp = client.get("/api/report/compare")
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_missing_b_param_returns_400(self, client, tmp_path, monkeypatch):
        _make_report(tmp_path, monkeypatch, "report_cmp_only_a")
        resp = client.get("/api/report/compare?a=report_cmp_only_a")
        assert resp.status_code == 400

    def test_nonexistent_a_returns_404(self, client, tmp_path, monkeypatch):
        _make_report(tmp_path, monkeypatch, "report_cmp_b1")
        resp = client.get("/api/report/compare?a=report_notfound&b=report_cmp_b1")
        assert resp.status_code == 404

    def test_nonexistent_b_returns_404(self, client, tmp_path, monkeypatch):
        _make_report(tmp_path, monkeypatch, "report_cmp_a1")
        resp = client.get("/api/report/compare?a=report_cmp_a1&b=report_notfound2")
        assert resp.status_code == 404

    def test_compare_two_reports_returns_200(self, client, tmp_path, monkeypatch):
        _make_report(tmp_path, monkeypatch, "report_cmp_x", sections=[
            ReportSection(title="Shared Section"),
            ReportSection(title="Only in X"),
        ])
        _make_report(tmp_path, monkeypatch, "report_cmp_y", sections=[
            ReportSection(title="Shared Section"),
            ReportSection(title="Only in Y"),
        ])
        resp = client.get("/api/report/compare?a=report_cmp_x&b=report_cmp_y")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        comparison = data["data"]
        assert "report_a" in comparison
        assert "report_b" in comparison
        diff = comparison["diff"]
        assert "Shared Section" in diff["sections_in_both"]
        assert "Only in X" in diff["sections_only_in_a"]
        assert "Only in Y" in diff["sections_only_in_b"]

    def test_compare_same_id_returns_tie(self, client, tmp_path, monkeypatch):
        _make_report(tmp_path, monkeypatch, "report_cmp_same")
        resp = client.get("/api/report/compare?a=report_cmp_same&b=report_cmp_same")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["data"]["diff"]["higher_evidence_score"] == "tie"
        assert data["data"]["diff"]["evidence_score_delta"] == 0.0

    def test_compare_delta_calculated_correctly(self, client, tmp_path, monkeypatch):
        # Report A: score 82, Report B: score 50 (no evidence)
        _make_report(tmp_path, monkeypatch, "report_cmp_high", with_evidence=True)
        _make_report(tmp_path, monkeypatch, "report_cmp_low", with_evidence=False)
        resp = client.get("/api/report/compare?a=report_cmp_high&b=report_cmp_low")
        assert resp.status_code == 200
        data = resp.get_json()
        diff = data["data"]["diff"]
        # B has no evidence (score=0), A has score=82 → delta = 0 - 82 = -82
        assert diff["evidence_score_delta"] < 0
        assert diff["higher_evidence_score"] == "report_cmp_high"


# ──────────────────────────────────────────────────────────────────────────────
# Feature 1: POST /api/report/<report_id>/retry endpoint
# ──────────────────────────────────────────────────────────────────────────────

class TestRetryReportEndpoint:
    def test_retry_nonexistent_returns_404(self, client):
        resp = client.post("/api/report/report_noexist_retry/retry")
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["success"] is False

    def test_retry_completed_report_returns_409(self, client, tmp_path, monkeypatch):
        _make_report(tmp_path, monkeypatch, "report_retry_done", status=ReportStatus.COMPLETED)
        resp = client.post("/api/report/report_retry_done/retry")
        assert resp.status_code == 409
        data = resp.get_json()
        assert data["success"] is False
        assert "already completed" in data["error"].lower()

    def test_retry_failed_report_returns_200(self, client, tmp_path, monkeypatch):
        """Retry a FAILED report: should accept and return task info."""
        _make_report(tmp_path, monkeypatch, "report_retry_failed", status=ReportStatus.FAILED)
        # We also need a simulation for the retry to find
        _make_simulation(tmp_path, monkeypatch, "sim_test")
        resp = client.post("/api/report/report_retry_failed/retry")
        assert resp.status_code == 200

    def test_retry_returns_task_id_and_report_id(self, client, tmp_path, monkeypatch):
        """When retry is accepted it returns task_id and report_id."""
        _make_report(tmp_path, monkeypatch, "report_retry_info", status=ReportStatus.FAILED)
        _make_simulation(tmp_path, monkeypatch, "sim_test")
        resp = client.post("/api/report/report_retry_info/retry")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "task_id" in data["data"]
        assert data["data"]["report_id"] == "report_retry_info"
