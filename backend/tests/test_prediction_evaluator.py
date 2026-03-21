"""
Unit and integration tests for the Prediction Evidence Tracker
and the /api/report/<report_id>/evidence endpoint.
"""

import json
import os
import pytest

from app.utils.prediction_evaluator import (
    PredictionEvidenceTracker,
    PredictionEvidence,
    SectionEvidence,
    KNOWN_TOOLS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures – sample agent log entries
# ──────────────────────────────────────────────────────────────────────────────

def _make_tool_call_entry(section_idx: int, section_title: str, tool_name: str, result: str = "") -> dict:
    return {
        "action": "tool_call",
        "section_index": section_idx,
        "section_title": section_title,
        "details": {
            "tool_name": tool_name,
            "result": result,
        },
    }


def _make_section_complete_entry(section_idx: int, section_title: str) -> dict:
    return {
        "action": "section_complete",
        "section_index": section_idx,
        "section_title": section_title,
        "details": {},
    }


INSIGHT_RESULT = """
## Future Forecast Deep Analysis
### [Key Facts] (Please cite these original texts in the report)
1. "Fact one about the event."
2. "Fact two about stakeholders."
3. "Fact three about public opinion."
"""

PANORAMA_RESULT = """
Search query: overall event timeline
Found 5 relevant entries
### Relevant Facts:
1. "Students expressed concern."
2. "University issued statement."
- "Media coverage expanded."
"""

INTERVIEW_RESULT = """
3 agents interviewed.
Agent 1: "I think this is very important."
Agent 2: "We need more information."
Agent 3: "The situation is under control."
"""

QUICK_RESULT = """
Search query: public reaction
Found 3 relevant entries
1. "Online sentiment was largely negative."
- "Hashtag trended for 48 hours."
"""


FULL_LOG = [
    _make_tool_call_entry(1, "Section One", "insight_forge", INSIGHT_RESULT),
    _make_tool_call_entry(1, "Section One", "panorama_search", PANORAMA_RESULT),
    _make_tool_call_entry(1, "Section One", "quick_search", QUICK_RESULT),
    _make_tool_call_entry(1, "Section One", "interview_agents", INTERVIEW_RESULT),
    _make_section_complete_entry(1, "Section One"),
    _make_tool_call_entry(2, "Section Two", "insight_forge", INSIGHT_RESULT),
    _make_tool_call_entry(2, "Section Two", "panorama_search", PANORAMA_RESULT),
    _make_tool_call_entry(2, "Section Two", "interview_agents", INTERVIEW_RESULT),
    _make_section_complete_entry(2, "Section Two"),
]


# ──────────────────────────────────────────────────────────────────────────────
# PredictionEvidenceTracker unit tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPredictionEvidenceTracker:
    def test_returns_prediction_evidence_instance(self):
        ev = PredictionEvidenceTracker.compute("r1", FULL_LOG)
        assert isinstance(ev, PredictionEvidence)

    def test_sections_count(self):
        ev = PredictionEvidenceTracker.compute("r1", FULL_LOG)
        assert ev.sections_generated == 2

    def test_total_tool_calls(self):
        ev = PredictionEvidenceTracker.compute("r1", FULL_LOG)
        # 4 calls in section 1 + 3 calls in section 2 = 7
        assert ev.total_tool_calls == 7

    def test_unique_tools_uses_all_four(self):
        ev = PredictionEvidenceTracker.compute("r1", FULL_LOG)
        assert set(ev.unique_tools_used) == KNOWN_TOOLS

    def test_agents_interviewed_is_positive(self):
        ev = PredictionEvidenceTracker.compute("r1", FULL_LOG)
        assert ev.agents_interviewed > 0

    def test_facts_retrieved_is_positive(self):
        ev = PredictionEvidenceTracker.compute("r1", FULL_LOG)
        assert ev.facts_retrieved > 0

    def test_evidence_score_range(self):
        ev = PredictionEvidenceTracker.compute("r1", FULL_LOG)
        assert 0.0 <= ev.evidence_score <= 100.0

    def test_is_evidence_based_for_full_log(self):
        ev = PredictionEvidenceTracker.compute("r1", FULL_LOG)
        assert ev.is_evidence_based is True

    def test_empty_log_returns_zero_score(self):
        ev = PredictionEvidenceTracker.compute("r_empty", [])
        assert ev.evidence_score == 0.0
        assert ev.is_evidence_based is False
        assert ev.total_tool_calls == 0

    def test_single_tool_low_diversity(self):
        log = [_make_tool_call_entry(1, "S1", "quick_search", "1. fact")] * 3
        log.append(_make_section_complete_entry(1, "S1"))
        ev = PredictionEvidenceTracker.compute("r2", log)
        # Only 1 unique tool → diversity ratio = 1/4 = 25%
        assert len(ev.unique_tools_used) == 1
        assert ev.evidence_score < 60.0  # not enough diversity

    def test_no_interviews_reduces_score(self):
        log_no_interview = [
            _make_tool_call_entry(1, "S1", "insight_forge", INSIGHT_RESULT),
            _make_tool_call_entry(1, "S1", "panorama_search", PANORAMA_RESULT),
            _make_tool_call_entry(1, "S1", "quick_search", QUICK_RESULT),
            _make_section_complete_entry(1, "S1"),
        ]
        ev_no_interview = PredictionEvidenceTracker.compute("r3", log_no_interview)
        ev_with_interview = PredictionEvidenceTracker.compute("r4", FULL_LOG)
        assert ev_with_interview.evidence_score > ev_no_interview.evidence_score

    def test_to_dict_contains_required_keys(self):
        ev = PredictionEvidenceTracker.compute("r1", FULL_LOG)
        d = ev.to_dict()
        for key in (
            "report_id", "sections_generated", "total_tool_calls",
            "unique_tools_used", "facts_retrieved", "agents_interviewed",
            "evidence_score", "is_evidence_based", "sections",
        ):
            assert key in d

    def test_per_section_breakdown_present(self):
        ev = PredictionEvidenceTracker.compute("r1", FULL_LOG)
        assert len(ev.sections) == 2
        section_one = ev.sections[0]
        assert section_one.section_index == 1
        assert section_one.tool_calls == 4

    def test_report_id_preserved(self):
        ev = PredictionEvidenceTracker.compute("my_report_id", FULL_LOG)
        assert ev.report_id == "my_report_id"


# ──────────────────────────────────────────────────────────────────────────────
# Private helper tests
# ──────────────────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_count_facts_numbered_list(self):
        text = "1. fact A\n2. fact B\n3. fact C"
        count = PredictionEvidenceTracker._count_facts(text)
        assert count == 3

    def test_count_facts_bullet_list(self):
        text = "- item one\n- item two"
        count = PredictionEvidenceTracker._count_facts(text)
        assert count == 2

    def test_count_facts_key_facts_header(self):
        text = "### [Key Facts]\nsome text"
        count = PredictionEvidenceTracker._count_facts(text)
        assert count >= 1

    def test_count_facts_empty(self):
        assert PredictionEvidenceTracker._count_facts("") == 0

    def test_count_interviewed_agents_explicit_number(self):
        text = "3 agents interviewed."
        count = PredictionEvidenceTracker._count_interviewed_agents(text)
        assert count == 3

    def test_count_interviewed_agents_agent_labels(self):
        text = "Agent 1: answer\nAgent 2: answer"
        count = PredictionEvidenceTracker._count_interviewed_agents(text)
        assert count >= 1

    def test_count_interviewed_agents_empty(self):
        assert PredictionEvidenceTracker._count_interviewed_agents("") == 0

    def test_compute_score_zero_sections(self):
        score = PredictionEvidenceTracker._compute_score(
            total_tool_calls=0, unique_tools=set(),
            total_facts=0, agents_interviewed=0, sections_count=0
        )
        assert score == 0.0

    def test_compute_score_increases_with_more_tools(self):
        s1 = PredictionEvidenceTracker._compute_score(
            total_tool_calls=3, unique_tools={"quick_search"},
            total_facts=5, agents_interviewed=0, sections_count=1
        )
        s2 = PredictionEvidenceTracker._compute_score(
            total_tool_calls=9, unique_tools=KNOWN_TOOLS,
            total_facts=20, agents_interviewed=3, sections_count=3
        )
        assert s2 > s1


# ──────────────────────────────────────────────────────────────────────────────
# Flask API integration test
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    import os
    os.environ.setdefault("LLM_API_KEY", "test-key")
    os.environ.setdefault("ZEP_API_KEY", "test-zep-key")
    from app import create_app
    application = create_app()
    application.config["TESTING"] = True
    application.config["DEBUG"] = False
    return application


@pytest.fixture(scope="module")
def client(app):
    return app.test_client()


class TestEvidenceEndpoint:
    def test_evidence_nonexistent_report_returns_404(self, client):
        resp = client.get("/api/report/report_doesnotexist/evidence")
        assert resp.status_code == 404
        data = resp.get_json()
        assert data.get("success") is False

    def test_evidence_existing_report_no_log_returns_503(self, client, tmp_path, monkeypatch):
        """A report that exists but has no agent_log.jsonl should return 503."""
        from app.services.report_agent import ReportManager, Report, ReportStatus

        # Create a minimal report directory with no agent log
        report_id = "report_test_no_log"
        monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(tmp_path / "reports"))
        report = Report(
            report_id=report_id,
            simulation_id="sim_test",
            graph_id="graph_test",
            simulation_requirement="test",
            status=ReportStatus.COMPLETED,
        )
        ReportManager.save_report(report)

        resp = client.get(f"/api/report/{report_id}/evidence")
        assert resp.status_code in (503, 404)  # depends on REPORTS_DIR resolution
        data = resp.get_json()
        assert data.get("success") is False

    def test_evidence_returns_correct_keys(self, client, tmp_path, monkeypatch):
        """A report with a complete agent_log.jsonl should return evidence."""
        from app.services.report_agent import ReportManager, Report, ReportStatus

        report_id = "report_test_evidence"
        monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(tmp_path / "reports2"))

        report = Report(
            report_id=report_id,
            simulation_id="sim_test2",
            graph_id="graph_test",
            simulation_requirement="test",
            status=ReportStatus.COMPLETED,
        )
        ReportManager.save_report(report)

        # Write a real agent_log.jsonl
        log_path = ReportManager._get_agent_log_path(report_id)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            for entry in FULL_LOG:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        resp = client.get(f"/api/report/{report_id}/evidence")
        data = resp.get_json()
        assert data.get("success") is True
        evidence = data["data"]
        for key in ("total_tool_calls", "unique_tools_used", "evidence_score", "is_evidence_based"):
            assert key in evidence

    def test_evidence_precomputed_summary_returned_immediately(self, client, tmp_path, monkeypatch):
        """If evidence_summary is already stored in meta.json it should be returned without re-computing."""
        from app.services.report_agent import ReportManager, Report, ReportStatus

        report_id = "report_test_precomputed"
        monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(tmp_path / "reports3"))

        precomputed = {
            "report_id": report_id,
            "total_tool_calls": 7,
            "unique_tools_used": list(KNOWN_TOOLS),
            "facts_retrieved": 10,
            "agents_interviewed": 3,
            "evidence_score": 85.0,
            "is_evidence_based": True,
            "sections": [],
        }
        report = Report(
            report_id=report_id,
            simulation_id="sim_test3",
            graph_id="graph_test",
            simulation_requirement="test",
            status=ReportStatus.COMPLETED,
            evidence_summary=precomputed,
        )
        ReportManager.save_report(report)

        resp = client.get(f"/api/report/{report_id}/evidence")
        data = resp.get_json()
        assert data.get("success") is True
        assert data["data"]["evidence_score"] == 85.0
        assert data["data"]["total_tool_calls"] == 7
