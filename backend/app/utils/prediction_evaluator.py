"""
Prediction Evidence Tracker

Analyses a completed report's ``agent_log.jsonl`` to produce concrete,
machine-readable metrics that demonstrate *how* MiroFish derives its
predictions.  Surfaces the evidence trail so that both developers and
end-users can verify that every prediction is grounded in actual
simulation data, not LLM fabrication.

Metrics computed
----------------
- total_tool_calls       – number of retrieval tool calls made across all sections
- unique_tools_used      – which of the four tools were actually called
- facts_retrieved        – lower-bound count of distinct facts returned by tools
- agents_interviewed     – number of distinct agents queried via interview_agents
- sections_generated     – number of report sections produced
- tool_calls_per_section – per-section breakdown
- evidence_score         – 0-100 composite score; ≥60 means "evidence-based"
- is_evidence_based      – True when evidence_score ≥ 60
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

KNOWN_TOOLS = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

# Minimum tool calls per section that we consider "well-supported"
_MIN_TOOL_CALLS_PER_SECTION = 3

# Score weights (must sum to 100)
_W_TOOL_CALLS = 30   # raw tool-call volume
_W_DIVERSITY = 30    # unique tool types used
_W_FACTS = 20        # facts retrieved
_W_INTERVIEW = 20    # whether agents were interviewed


@dataclass
class SectionEvidence:
    """Evidence metrics for a single report section."""

    section_index: int
    section_title: str
    tool_calls: int = 0
    tools_used: List[str] = field(default_factory=list)
    facts_count: int = 0
    agents_interviewed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_index": self.section_index,
            "section_title": self.section_title,
            "tool_calls": self.tool_calls,
            "tools_used": sorted(set(self.tools_used)),
            "facts_count": self.facts_count,
            "agents_interviewed": self.agents_interviewed,
        }


@dataclass
class PredictionEvidence:
    """Aggregate prediction evidence for a complete report."""

    report_id: str
    sections_generated: int
    total_tool_calls: int
    unique_tools_used: List[str]
    facts_retrieved: int
    agents_interviewed: int
    evidence_score: float
    is_evidence_based: bool
    sections: List[SectionEvidence] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "sections_generated": self.sections_generated,
            "total_tool_calls": self.total_tool_calls,
            "unique_tools_used": sorted(self.unique_tools_used),
            "facts_retrieved": self.facts_retrieved,
            "agents_interviewed": self.agents_interviewed,
            "evidence_score": round(self.evidence_score, 1),
            "is_evidence_based": self.is_evidence_based,
            "sections": [s.to_dict() for s in self.sections],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Evidence tracker
# ──────────────────────────────────────────────────────────────────────────────

class PredictionEvidenceTracker:
    """Parse ``agent_log.jsonl`` entries and compute prediction evidence metrics.

    Usage::

        with open(log_path) as f:
            entries = [json.loads(line) for line in f if line.strip()]
        evidence = PredictionEvidenceTracker.compute(report_id, entries)
    """

    @staticmethod
    def compute(report_id: str, log_entries: List[Dict[str, Any]]) -> PredictionEvidence:
        """Compute evidence metrics from parsed log entries.

        Args:
            report_id: Identifier of the report.
            log_entries: List of dicts parsed from ``agent_log.jsonl``.

        Returns:
            A :class:`PredictionEvidence` instance.
        """
        # Accumulate per-section data using a dict keyed by section_index
        sections: Dict[int, SectionEvidence] = {}

        for entry in log_entries:
            action = entry.get("action", "")
            details = entry.get("details", {})
            section_idx = entry.get("section_index") or 0
            section_title = entry.get("section_title") or ""

            # Ensure a SectionEvidence bucket exists for this section
            if section_idx not in sections and section_idx > 0:
                sections[section_idx] = SectionEvidence(
                    section_index=section_idx,
                    section_title=section_title or f"Section {section_idx}",
                )

            if action == "tool_call":
                tool_name = details.get("tool_name", "")
                sec = sections.get(section_idx)
                if sec:
                    sec.tool_calls += 1
                    if tool_name:
                        sec.tools_used.append(tool_name)

                    # Count facts in tool result (rough heuristic: numbered lines or "Relevant Facts" headers)
                    result_text = details.get("result", "")
                    if result_text:
                        sec.facts_count += PredictionEvidenceTracker._count_facts(result_text)

                    # Count agents from interview results
                    if tool_name == "interview_agents":
                        sec.agents_interviewed += PredictionEvidenceTracker._count_interviewed_agents(result_text)

            elif action == "section_complete" or action == "section_content":
                # Update the title in case it was set on a later log entry
                sec = sections.get(section_idx)
                if sec and section_title:
                    sec.section_title = section_title

        sections_list = sorted(sections.values(), key=lambda s: s.section_index)

        # Aggregate
        total_tool_calls = sum(s.tool_calls for s in sections_list)
        unique_tools: Set[str] = set()
        for s in sections_list:
            unique_tools.update(t for t in s.tools_used if t in KNOWN_TOOLS)
        total_facts = sum(s.facts_count for s in sections_list)
        total_agents = sum(s.agents_interviewed for s in sections_list)

        score = PredictionEvidenceTracker._compute_score(
            total_tool_calls=total_tool_calls,
            unique_tools=unique_tools,
            total_facts=total_facts,
            agents_interviewed=total_agents,
            sections_count=len(sections_list),
        )

        return PredictionEvidence(
            report_id=report_id,
            sections_generated=len(sections_list),
            total_tool_calls=total_tool_calls,
            unique_tools_used=list(unique_tools),
            facts_retrieved=total_facts,
            agents_interviewed=total_agents,
            evidence_score=score,
            is_evidence_based=score >= 60.0,
            sections=sections_list,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _count_facts(text: str) -> int:
        """Estimate the number of distinct facts in a tool result string.

        Counts lines that start with a digit followed by a period/dot (numbered
        lists), or lines that start with '- ' or '* ' (bullet lists).  Also
        counts "Key Facts" section markers as an extra 1.
        """
        count = 0
        for line in text.splitlines():
            stripped = line.strip()
            if re.match(r'^\d+[\.\)]\s+\S', stripped):
                count += 1
            elif stripped.startswith("- ") or stripped.startswith("* "):
                count += 1
        # Bonus for well-structured InsightForge output
        if "Key Facts" in text or "Relevant Facts" in text:
            count = max(count, 1)
        return count

    @staticmethod
    def _count_interviewed_agents(result_text: str) -> int:
        """Estimate number of agents interviewed from the result text.

        Looks for markers like "Agent:", "@agent_name", "interview" lines, or
        explicit count patterns such as "3 agents interviewed".
        """
        if not result_text:
            return 0

        # Pattern: "X agents" or "X agent" followed by a context word
        m = re.search(r'(\d+)\s+agents?\b', result_text, re.IGNORECASE)
        if m:
            return int(m.group(1))

        # Fallback: count "Agent:" occurrences
        count = len(re.findall(r'\bAgent\s*\d*\s*:', result_text, re.IGNORECASE))
        return max(count, 1) if result_text.strip() else 0

    @staticmethod
    def _compute_score(
        total_tool_calls: int,
        unique_tools: Set[str],
        total_facts: int,
        agents_interviewed: int,
        sections_count: int,
    ) -> float:
        """Compute the 0-100 evidence score.

        Each component is normalised to [0, 1] and weighted:

        - Tool call volume  (30 pts): ≥3 calls per section = full marks
        - Tool diversity    (30 pts): using all 4 known tools = full marks
        - Facts retrieved   (20 pts): ≥20 facts = full marks (logarithmic)
        - Agent interviews  (20 pts): ≥1 interview = full marks
        """
        if sections_count == 0:
            return 0.0

        # 1. Tool call volume
        calls_per_section = total_tool_calls / sections_count
        volume_ratio = min(calls_per_section / _MIN_TOOL_CALLS_PER_SECTION, 1.0)
        volume_score = volume_ratio * _W_TOOL_CALLS

        # 2. Tool diversity
        diversity_ratio = len(unique_tools) / len(KNOWN_TOOLS)
        diversity_score = diversity_ratio * _W_DIVERSITY

        # 3. Facts retrieved (logarithmic scale: 0→0, 5→50%, 20→100%)
        import math
        facts_ratio = min(math.log(total_facts + 1) / math.log(21), 1.0)
        facts_score = facts_ratio * _W_FACTS

        # 4. Agent interviews
        interview_score = _W_INTERVIEW if agents_interviewed > 0 else 0.0

        return volume_score + diversity_score + facts_score + interview_score
