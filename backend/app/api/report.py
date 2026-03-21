"""
Report API routes
=================

Provides all endpoints related to simulation analysis reports.

Report lifecycle
----------------
1. ``POST /api/report/generate``
   Kick off an async generation task.  Returns ``task_id`` + ``report_id``
   immediately so the client can poll for progress without blocking.

2. ``GET  /api/report/generate/status``
   Poll task progress; returns the current stage / percentage.

3. ``GET  /api/report/<report_id>``
   Retrieve the completed report JSON (outline, sections, evidence summary).

New roadmap endpoints
---------------------
- ``POST   /api/report/<report_id>/retry``
   Reset a FAILED or PENDING report and re-run generation in a background
   thread.  Returns 409 if the report is already COMPLETED.

- ``GET    /api/report/<report_id>/export/html``
   Render the report as a self-contained HTML page (stdlib only — no extra
   dependencies) and return it as a file attachment.  An *evidence badge*
   is embedded when ``evidence_summary`` is available.

- ``DELETE /api/report/<report_id>``
   Permanently remove a report from disk and memory.

- ``GET    /api/report/compare?a=<id>&b=<id>``
   Side-by-side comparison of two reports: evidence scores, section outlines,
   and the symmetric difference between their section title sets.

- ``GET    /api/report/<report_id>/evidence``
   Machine-readable evidence metrics (tool calls, facts, agents interviewed,
   composite score 0–100) parsed from ``agent_log.jsonl``.

Streaming / logging
-------------------
- ``GET  /api/report/<report_id>/agent-log``        — full JSONL agent log
- ``GET  /api/report/<report_id>/agent-log/stream`` — SSE stream of agent log
- ``GET  /api/report/<report_id>/console-log``      — human-readable console
- ``GET  /api/report/<report_id>/console-log/stream``— SSE stream of console

Conversation
------------
- ``POST /api/report/chat``  — continue a multi-turn conversation with the
  report agent; the agent can call retrieval tools to answer follow-up questions.
"""

import os
import threading
from flask import request, jsonify, send_file

from . import report_bp
from ..config import Config
from ..services.report_agent import ReportAgent, ReportManager, ReportStatus
from ..services.simulation_manager import SimulationManager
from ..models.project import ProjectManager
from ..models.task import TaskManager, TaskStatus
from ..utils.logger import get_logger
from ..utils.response import error_response
from ..utils.validators import GenerateReportRequest, parse_request

logger = get_logger('mirofish.api.report')


# ============== Report generation endpoints ==============

@report_bp.route('/generate', methods=['POST'])
def generate_report():
    """
    Generate a simulation analysis report (async task)
    
    This is a time-consuming operation; the endpoint returns a task_id immediately.
    Use GET /api/report/generate/status to check progress.
    
    Request (JSON):
        {
            "simulation_id": "sim_xxxx",    // required, simulation ID
            "force_regenerate": false        // optional, force regeneration
        }
    
    Returns:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "task_id": "task_xxxx",
                "status": "generating",
                "message": "Report generation task started"
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        validated, err = parse_request(GenerateReportRequest, data)
        if err:
            return jsonify({"success": False, "error": err}), 400
        
        simulation_id = validated.simulation_id
        force_regenerate = validated.force_regenerate
        
        # Get simulation information
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        
        if not state:
            return jsonify({
                "success": False,
                "error": f"Simulation not found: {simulation_id}"
            }), 404
        
        # Check whether a report already exists
        if not force_regenerate:
            existing_report = ReportManager.get_report_by_simulation(simulation_id)
            if existing_report and existing_report.status == ReportStatus.COMPLETED:
                return jsonify({
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "report_id": existing_report.report_id,
                        "status": "completed",
                        "message": "Report already exists",
                        "already_generated": True
                    }
                })
        
        # Get project information
        project = ProjectManager.get_project(state.project_id)
        if not project:
            return jsonify({
                "success": False,
                "error": f"Project not found: {state.project_id}"
            }), 404
        
        graph_id = state.graph_id or project.graph_id
        if not graph_id:
            return jsonify({
                "success": False,
                "error": "Graph ID is missing; please ensure the graph has been built"
            }), 400
        
        simulation_requirement = project.simulation_requirement
        if not simulation_requirement:
            return jsonify({
                "success": False,
                "error": "Simulation requirement description is missing"
            }), 400
        
        # Pre-generate report_id so it can be returned to the frontend immediately
        import uuid
        report_id = f"report_{uuid.uuid4().hex[:12]}"
        
        # Create async task
        task_manager = TaskManager()
        task_id = task_manager.create_task(
            task_type="report_generate",
            metadata={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "report_id": report_id
            }
        )
        
        # Define background task
        def run_generate():
            try:
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.PROCESSING,
                    progress=0,
                    message="Initializing Report Agent..."
                )
                
                # Create Report Agent
                agent = ReportAgent(
                    graph_id=graph_id,
                    simulation_id=simulation_id,
                    simulation_requirement=simulation_requirement
                )
                
                # Progress callback
                def progress_callback(stage, progress, message):
                    task_manager.update_task(
                        task_id,
                        progress=progress,
                        message=f"[{stage}] {message}"
                    )
                
                # Generate report (pass the pre-generated report_id)
                report = agent.generate_report(
                    progress_callback=progress_callback,
                    report_id=report_id
                )
                
                # Save report
                ReportManager.save_report(report)
                
                if report.status == ReportStatus.COMPLETED:
                    task_manager.complete_task(
                        task_id,
                        result={
                            "report_id": report.report_id,
                            "simulation_id": simulation_id,
                            "status": "completed"
                        }
                    )
                else:
                    task_manager.fail_task(task_id, report.error or "Report generation failed")
                
            except Exception as e:
                logger.error(f"Report generation failed: {str(e)}")
                task_manager.fail_task(task_id, str(e))
        
        # Start background thread
        thread = threading.Thread(target=run_generate, daemon=True)
        thread.start()
        
        return jsonify({
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "report_id": report_id,
                "task_id": task_id,
                "status": "generating",
                "message": "Report generation task started; use /api/report/generate/status to check progress",
                "already_generated": False
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to start report generation task: {str(e)}")
        return jsonify(error_response(str(e), 500, e)), 500


@report_bp.route('/generate/status', methods=['POST'])
def get_generate_status():
    """
    Query report generation task progress
    
    Request (JSON):
        {
            "task_id": "task_xxxx",         // optional, task_id returned by generate
            "simulation_id": "sim_xxxx"     // optional, simulation ID
        }
    
    Returns:
        {
            "success": true,
            "data": {
                "task_id": "task_xxxx",
                "status": "processing|completed|failed",
                "progress": 45,
                "message": "..."
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        task_id = data.get('task_id')
        simulation_id = data.get('simulation_id')
        
        # If simulation_id is provided, check whether a completed report exists first
        if simulation_id:
            existing_report = ReportManager.get_report_by_simulation(simulation_id)
            if existing_report and existing_report.status == ReportStatus.COMPLETED:
                return jsonify({
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "report_id": existing_report.report_id,
                        "status": "completed",
                        "progress": 100,
                        "message": "Report has been generated",
                        "already_completed": True
                    }
                })
        
        if not task_id:
            return jsonify({
                "success": False,
                "error": "Please provide task_id or simulation_id"
            }), 400
        
        task_manager = TaskManager()
        task = task_manager.get_task(task_id)
        
        if not task:
            return jsonify({
                "success": False,
                "error": f"Task not found: {task_id}"
            }), 404
        
        return jsonify({
            "success": True,
            "data": task.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Failed to query task status: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============== Report retrieval endpoints ==============

@report_bp.route('/<report_id>/evidence', methods=['GET'])
def get_report_evidence(report_id: str):
    """
    Get prediction evidence summary for a report.

    Returns concrete metrics proving that the prediction report is grounded
    in actual simulation data rather than LLM hallucination:

    - ``total_tool_calls``   – total retrieval tool calls made
    - ``unique_tools_used``  – which tools were used (diversity indicator)
    - ``facts_retrieved``    – lower-bound count of facts returned by tools
    - ``agents_interviewed`` – number of agents queried for first-person perspectives
    - ``evidence_score``     – 0–100 composite score (≥ 60 means evidence-based)
    - ``is_evidence_based``  – True when evidence_score ≥ 60
    - ``sections``           – per-section breakdown

    Returns 404 when the report does not exist and 503 when the agent log is
    not yet available (report still generating).
    """
    try:
        report = ReportManager.get_report(report_id)
        if not report:
            return jsonify({
                "success": False,
                "error": f"Report not found: {report_id}"
            }), 404

        # If already computed and saved, return it immediately
        if report.evidence_summary:
            return jsonify({
                "success": True,
                "data": report.evidence_summary
            })

        # Otherwise, attempt on-demand computation from the agent log
        summary = ReportManager.compute_evidence_summary(report_id)
        if summary is None:
            return jsonify({
                "success": False,
                "error": "Evidence log not yet available; report may still be generating",
            }), 503

        return jsonify({
            "success": True,
            "data": summary
        })

    except Exception as e:
        logger.error(f"Failed to get prediction evidence: {str(e)}")
        return jsonify(error_response(str(e), 500, e)), 500


@report_bp.route('/<report_id>/retry', methods=['POST'])
def retry_report(report_id: str):
    """
    Retry generation of a failed or incomplete report.

    Clears the error/FAILED state and re-runs the full generation pipeline
    in a background thread.  Returns 404 if the report does not exist; use
    ``POST /generate`` with a ``simulation_id`` to create a brand-new report.
    Returns 409 if the report is already COMPLETED.

    Returns the same task_id / report_id pair as ``POST /generate`` so that
    the caller can poll ``POST /generate/status`` for progress.
    """
    try:
        report = ReportManager.get_report(report_id)
        if not report:
            return jsonify({
                "success": False,
                "error": f"Report not found: {report_id}"
            }), 404

        if report.status == ReportStatus.COMPLETED:
            return jsonify({
                "success": False,
                "error": "Report is already completed. Use force_regenerate via POST /generate if you want a fresh copy."
            }), 409

        # Retrieve the simulation so we can reconstruct the agent
        manager = SimulationManager()
        state = manager.get_simulation(report.simulation_id)
        if not state:
            return jsonify({
                "success": False,
                "error": f"Simulation not found: {report.simulation_id}"
            }), 404

        project = ProjectManager.get_project(state.project_id)
        simulation_requirement = (
            project.simulation_requirement if project else report.simulation_requirement
        )
        graph_id = report.graph_id

        # Reset report state
        report.status = ReportStatus.PENDING
        report.error = None
        report.markdown_content = ""
        report.evidence_summary = None
        report.completed_at = ""
        ReportManager.save_report(report)

        # Create task
        task_manager = TaskManager()
        task_id = task_manager.create_task(
            task_type="report_generate",
            metadata={
                "simulation_id": report.simulation_id,
                "graph_id": graph_id,
                "report_id": report_id,
                "is_retry": True,
            }
        )

        def run_retry():
            try:
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.PROCESSING,
                    progress=0,
                    message="Retrying report generation…"
                )
                agent = ReportAgent(
                    graph_id=graph_id,
                    simulation_id=report.simulation_id,
                    simulation_requirement=simulation_requirement,
                )

                def progress_callback(stage, progress, message):
                    task_manager.update_task(
                        task_id, progress=progress,
                        message=f"[{stage}] {message}"
                    )

                result = agent.generate_report(
                    progress_callback=progress_callback,
                    report_id=report_id,
                )
                ReportManager.save_report(result)

                if result.status == ReportStatus.COMPLETED:
                    task_manager.complete_task(task_id, result={
                        "report_id": report_id,
                        "simulation_id": report.simulation_id,
                        "status": "completed",
                    })
                else:
                    task_manager.fail_task(task_id, result.error or "Retry failed")
            except Exception as exc:
                logger.error(f"Report retry failed: {exc}")
                task_manager.fail_task(task_id, str(exc))

        threading.Thread(target=run_retry, daemon=True).start()

        return jsonify({
            "success": True,
            "data": {
                "report_id": report_id,
                "task_id": task_id,
                "status": "retrying",
                "message": "Retry started; poll POST /api/report/generate/status for progress"
            }
        })

    except Exception as e:
        logger.error(f"Failed to start report retry: {str(e)}")
        return jsonify(error_response(str(e), 500, e)), 500


@report_bp.route('/<report_id>/export/html', methods=['GET'])
def export_report_html(report_id: str):
    """
    Export a completed report as a self-contained HTML page.

    The HTML includes a minimal CSS stylesheet and, when available, an
    evidence badge summarising the prediction quality metrics
    (evidence_score, tool_calls, facts, agents_interviewed).

    Returns a ``text/html`` file attachment.  Returns 404 when the report
    does not exist and 400 when the report has not been completed yet.
    """
    try:
        report = ReportManager.get_report(report_id)
        if not report:
            return jsonify({
                "success": False,
                "error": f"Report not found: {report_id}"
            }), 404

        if report.status != ReportStatus.COMPLETED:
            return jsonify({
                "success": False,
                "error": f"Report is not complete yet (status: {report.status.value})"
            }), 400

        html_content = ReportManager.render_html(report)

        import tempfile
        from flask import after_this_request

        with tempfile.NamedTemporaryFile(mode='w', suffix='.html',
                                         delete=False, encoding='utf-8') as f:
            f.write(html_content)
            temp_path = f.name

        @after_this_request
        def _remove_temp(response):
            try:
                os.remove(temp_path)
            except OSError:
                pass
            return response

        return send_file(
            temp_path,
            as_attachment=True,
            download_name=f"{report_id}.html",
            mimetype="text/html"
        )

    except Exception as e:
        logger.error(f"Failed to export report as HTML: {str(e)}")
        return jsonify(error_response(str(e), 500, e)), 500


@report_bp.route('/compare', methods=['GET'])
def compare_reports():
    """
    Compare two reports side-by-side.

    Query parameters:
        a  – first report ID  (required)
        b  – second report ID (required)

    Returns a structured comparison including:
    - Metadata diff (title, sections count, completion times)
    - Evidence score comparison (delta and winner)
    - Section-level outline diff (titles present in each report)

    Returns 400 when either ID is missing and 404 when a report is not found.
    """
    try:
        id_a = request.args.get("a", "").strip()
        id_b = request.args.get("b", "").strip()

        if not id_a or not id_b:
            return jsonify({
                "success": False,
                "error": "Both query parameters 'a' and 'b' (report IDs) are required"
            }), 400

        report_a = ReportManager.get_report(id_a)
        if not report_a:
            return jsonify({
                "success": False,
                "error": f"Report not found: {id_a}"
            }), 404

        report_b = ReportManager.get_report(id_b)
        if not report_b:
            return jsonify({
                "success": False,
                "error": f"Report not found: {id_b}"
            }), 404

        def _outline_titles(report):
            if report.outline:
                return [s.title for s in report.outline.sections]
            return []

        def _ev(report):
            if report.evidence_summary:
                return report.evidence_summary
            # Fallback: compute on demand from agent_log.jsonl for older reports
            computed = ReportManager.compute_evidence_summary(report.report_id)
            if computed:
                report.evidence_summary = computed
                ReportManager.save_report(report)
            return computed or {}

        titles_a = _outline_titles(report_a)
        titles_b = _outline_titles(report_b)
        set_a = set(titles_a)
        set_b = set(titles_b)

        score_a = _ev(report_a).get("evidence_score", 0.0)
        score_b = _ev(report_b).get("evidence_score", 0.0)
        delta = round(score_b - score_a, 1)
        if score_a > score_b:
            winner = id_a
        elif score_b > score_a:
            winner = id_b
        else:
            winner = "tie"

        comparison = {
            "report_a": {
                "report_id": id_a,
                "title": report_a.outline.title if report_a.outline else "",
                "status": report_a.status.value,
                "sections_count": len(titles_a),
                "section_titles": titles_a,
                "evidence_score": score_a,
                "is_evidence_based": _ev(report_a).get("is_evidence_based", False),
                "total_tool_calls": _ev(report_a).get("total_tool_calls", 0),
                "facts_retrieved": _ev(report_a).get("facts_retrieved", 0),
                "agents_interviewed": _ev(report_a).get("agents_interviewed", 0),
                "completed_at": report_a.completed_at,
            },
            "report_b": {
                "report_id": id_b,
                "title": report_b.outline.title if report_b.outline else "",
                "status": report_b.status.value,
                "sections_count": len(titles_b),
                "section_titles": titles_b,
                "evidence_score": score_b,
                "is_evidence_based": _ev(report_b).get("is_evidence_based", False),
                "total_tool_calls": _ev(report_b).get("total_tool_calls", 0),
                "facts_retrieved": _ev(report_b).get("facts_retrieved", 0),
                "agents_interviewed": _ev(report_b).get("agents_interviewed", 0),
                "completed_at": report_b.completed_at,
            },
            "diff": {
                "evidence_score_delta": delta,
                "higher_evidence_score": winner,
                "sections_only_in_a": sorted(set_a - set_b),
                "sections_only_in_b": sorted(set_b - set_a),
                "sections_in_both": sorted(set_a & set_b),
            },
        }

        return jsonify({
            "success": True,
            "data": comparison
        })

    except Exception as e:
        logger.error(f"Failed to compare reports: {str(e)}")
        return jsonify(error_response(str(e), 500, e)), 500


@report_bp.route('/<report_id>', methods=['GET'])
def get_report(report_id: str):
    """
    Get report details
    
    Returns:
        {
            "success": true,
            "data": {
                "report_id": "report_xxxx",
                "simulation_id": "sim_xxxx",
                "status": "completed",
                "outline": {...},
                "markdown_content": "...",
                "created_at": "...",
                "completed_at": "..."
            }
        }
    """
    try:
        report = ReportManager.get_report(report_id)
        
        if not report:
            return jsonify({
                "success": False,
                "error": f"Report not found: {report_id}"
            }), 404
        
        return jsonify({
            "success": True,
            "data": report.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Failed to get report: {str(e)}")
        return jsonify(error_response(str(e), 500, e)), 500


@report_bp.route('/by-simulation/<simulation_id>', methods=['GET'])
def get_report_by_simulation(simulation_id: str):
    """
    Get a report by simulation ID
    
    Returns:
        {
            "success": true,
            "data": {
                "report_id": "report_xxxx",
                ...
            }
        }
    """
    try:
        report = ReportManager.get_report_by_simulation(simulation_id)
        
        if not report:
            return jsonify({
                "success": False,
                "error": f"No report found for this simulation: {simulation_id}",
                "has_report": False
            }), 404
        
        return jsonify({
            "success": True,
            "data": report.to_dict(),
            "has_report": True
        })
        
    except Exception as e:
        logger.error(f"Failed to get report: {str(e)}")
        return jsonify(error_response(str(e), 500, e)), 500


@report_bp.route('/list', methods=['GET'])
def list_reports():
    """
    List all reports
    
    Query parameters:
        simulation_id: Filter by simulation ID (optional)
        limit: Maximum number of results to return (default 50)
    
    Returns:
        {
            "success": true,
            "data": [...],
            "count": 10
        }
    """
    try:
        simulation_id = request.args.get('simulation_id')
        limit = request.args.get('limit', 50, type=int)
        
        reports = ReportManager.list_reports(
            simulation_id=simulation_id,
            limit=limit
        )
        
        return jsonify({
            "success": True,
            "data": [r.to_dict() for r in reports],
            "count": len(reports)
        })
        
    except Exception as e:
        logger.error(f"Failed to list reports: {str(e)}")
        return jsonify(error_response(str(e), 500, e)), 500


@report_bp.route('/<report_id>/download', methods=['GET'])
def download_report(report_id: str):
    """
    Download a report (Markdown format)
    
    Returns a Markdown file
    """
    try:
        report = ReportManager.get_report(report_id)
        
        if not report:
            return jsonify({
                "success": False,
                "error": f"Report not found: {report_id}"
            }), 404
        
        md_path = ReportManager._get_report_markdown_path(report_id)
        
        if not os.path.exists(md_path):
            # If the MD file doesn't exist, generate a temporary one
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write(report.markdown_content)
                temp_path = f.name
            
            return send_file(
                temp_path,
                as_attachment=True,
                download_name=f"{report_id}.md"
            )
        
        return send_file(
            md_path,
            as_attachment=True,
            download_name=f"{report_id}.md"
        )
        
    except Exception as e:
        logger.error(f"Failed to download report: {str(e)}")
        return jsonify(error_response(str(e), 500, e)), 500


@report_bp.route('/<report_id>', methods=['DELETE'])
def delete_report(report_id: str):
    """Delete a report"""
    try:
        success = ReportManager.delete_report(report_id)
        
        if not success:
            return jsonify({
                "success": False,
                "error": f"Report not found: {report_id}"
            }), 404
        
        return jsonify({
            "success": True,
            "message": f"Report deleted: {report_id}"
        })
        
    except Exception as e:
        logger.error(f"Failed to delete report: {str(e)}")
        return jsonify(error_response(str(e), 500, e)), 500


# ============== Report Agent conversation endpoint ==============

@report_bp.route('/chat', methods=['POST'])
def chat_with_report_agent():
    """
    Chat with the Report Agent
    
    The Report Agent can autonomously call retrieval tools during the conversation to answer questions
    
    Request (JSON):
        {
            "simulation_id": "sim_xxxx",            // required, simulation ID
            "message": "Please explain the sentiment trend",  // required, user message
            "chat_history": [                       // optional, conversation history
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }
    
    Returns:
        {
            "success": true,
            "data": {
                "response": "Agent reply...",
                "tool_calls": [list of tools called],
                "sources": [information sources]
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        simulation_id = data.get('simulation_id')
        message = data.get('message')
        chat_history = data.get('chat_history', [])
        
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Please provide simulation_id"
            }), 400
        
        if not message:
            return jsonify({
                "success": False,
                "error": "Please provide message"
            }), 400
        
        # Get simulation and project information
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        
        if not state:
            return jsonify({
                "success": False,
                "error": f"Simulation not found: {simulation_id}"
            }), 404
        
        project = ProjectManager.get_project(state.project_id)
        if not project:
            return jsonify({
                "success": False,
                "error": f"Project not found: {state.project_id}"
            }), 404
        
        graph_id = state.graph_id or project.graph_id
        if not graph_id:
            return jsonify({
                "success": False,
                "error": "Graph ID is missing"
            }), 400
        
        simulation_requirement = project.simulation_requirement or ""
        
        # Create Agent and start conversation
        agent = ReportAgent(
            graph_id=graph_id,
            simulation_id=simulation_id,
            simulation_requirement=simulation_requirement
        )
        
        result = agent.chat(message=message, chat_history=chat_history)
        
        return jsonify({
            "success": True,
            "data": result
        })
        
    except Exception as e:
        logger.error(f"Conversation failed: {str(e)}")
        return jsonify(error_response(str(e), 500, e)), 500


# ============== Report progress and section-by-section endpoints ==============

@report_bp.route('/<report_id>/progress', methods=['GET'])
def get_report_progress(report_id: str):
    """
    Get report generation progress (real-time)
    
    Returns:
        {
            "success": true,
            "data": {
                "status": "generating",
                "progress": 45,
                "message": "Generating section: Key Findings",
                "current_section": "Key Findings",
                "completed_sections": ["Executive Summary", "Simulation Background"],
                "updated_at": "2025-12-09T..."
            }
        }
    """
    try:
        progress = ReportManager.get_progress(report_id)
        
        if not progress:
            return jsonify({
                "success": False,
                "error": f"Report not found or progress information unavailable: {report_id}"
            }), 404
        
        return jsonify({
            "success": True,
            "data": progress
        })
        
    except Exception as e:
        logger.error(f"Failed to get report progress: {str(e)}")
        return jsonify(error_response(str(e), 500, e)), 500


@report_bp.route('/<report_id>/sections', methods=['GET'])
def get_report_sections(report_id: str):
    """
    Get the list of generated sections (section-by-section output)
    
    The frontend can poll this endpoint to get generated section content
    without waiting for the entire report to finish.
    
    Returns:
        {
            "success": true,
            "data": {
                "report_id": "report_xxxx",
                "sections": [
                    {
                        "filename": "section_01.md",
                        "section_index": 1,
                        "content": "## Executive Summary\\n\\n..."
                    },
                    ...
                ],
                "total_sections": 3,
                "is_complete": false
            }
        }
    """
    try:
        sections = ReportManager.get_generated_sections(report_id)
        
        # Get report status
        report = ReportManager.get_report(report_id)
        is_complete = report is not None and report.status == ReportStatus.COMPLETED
        
        return jsonify({
            "success": True,
            "data": {
                "report_id": report_id,
                "sections": sections,
                "total_sections": len(sections),
                "is_complete": is_complete
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get section list: {str(e)}")
        return jsonify(error_response(str(e), 500, e)), 500


@report_bp.route('/<report_id>/section/<int:section_index>', methods=['GET'])
def get_single_section(report_id: str, section_index: int):
    """
    Get a single section's content
    
    Returns:
        {
            "success": true,
            "data": {
                "filename": "section_01.md",
                "content": "## Executive Summary\\n\\n..."
            }
        }
    """
    try:
        section_path = ReportManager._get_section_path(report_id, section_index)
        
        if not os.path.exists(section_path):
            return jsonify({
                "success": False,
                "error": f"Section not found: section_{section_index:02d}.md"
            }), 404
        
        with open(section_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return jsonify({
            "success": True,
            "data": {
                "filename": f"section_{section_index:02d}.md",
                "section_index": section_index,
                "content": content
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get section content: {str(e)}")
        return jsonify(error_response(str(e), 500, e)), 500


# ============== Report status check endpoint ==============

@report_bp.route('/check/<simulation_id>', methods=['GET'])
def check_report_status(simulation_id: str):
    """
    Check whether a simulation has a report and its status
    
    Used by the frontend to determine whether to unlock the Interview feature
    
    Returns:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "has_report": true,
                "report_status": "completed",
                "report_id": "report_xxxx",
                "interview_unlocked": true
            }
        }
    """
    try:
        report = ReportManager.get_report_by_simulation(simulation_id)
        
        has_report = report is not None
        report_status = report.status.value if report else None
        report_id = report.report_id if report else None
        
        # Interview is only unlocked once the report is completed
        interview_unlocked = has_report and report.status == ReportStatus.COMPLETED
        
        return jsonify({
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "has_report": has_report,
                "report_status": report_status,
                "report_id": report_id,
                "interview_unlocked": interview_unlocked
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to check report status: {str(e)}")
        return jsonify(error_response(str(e), 500, e)), 500


# ============== Agent log endpoints ==============

@report_bp.route('/<report_id>/agent-log', methods=['GET'])
def get_agent_log(report_id: str):
    """
    Get the detailed execution log of the Report Agent
    
    Retrieve each step of the report generation process in real time, including:
    - Report start, planning start/complete
    - Each section's start, tool calls, LLM responses, and completion
    - Report completion or failure
    
    Query parameters:
        from_line: Line number to start reading from (optional, default 0, for incremental retrieval)
    
    Returns:
        {
            "success": true,
            "data": {
                "logs": [
                    {
                        "timestamp": "2025-12-13T...",
                        "elapsed_seconds": 12.5,
                        "report_id": "report_xxxx",
                        "action": "tool_call",
                        "stage": "generating",
                        "section_title": "Executive Summary",
                        "section_index": 1,
                        "details": {
                            "tool_name": "insight_forge",
                            "parameters": {...},
                            ...
                        }
                    },
                    ...
                ],
                "total_lines": 25,
                "from_line": 0,
                "has_more": false
            }
        }
    """
    try:
        from_line = request.args.get('from_line', 0, type=int)
        
        log_data = ReportManager.get_agent_log(report_id, from_line=from_line)
        
        return jsonify({
            "success": True,
            "data": log_data
        })
        
    except Exception as e:
        logger.error(f"Failed to get Agent log: {str(e)}")
        return jsonify(error_response(str(e), 500, e)), 500


@report_bp.route('/<report_id>/agent-log/stream', methods=['GET'])
def stream_agent_log(report_id: str):
    """
    Get the complete Agent log (all at once)
    
    Returns:
        {
            "success": true,
            "data": {
                "logs": [...],
                "count": 25
            }
        }
    """
    try:
        logs = ReportManager.get_agent_log_stream(report_id)
        
        return jsonify({
            "success": True,
            "data": {
                "logs": logs,
                "count": len(logs)
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get Agent log: {str(e)}")
        return jsonify(error_response(str(e), 500, e)), 500


# ============== Console log endpoints ==============

@report_bp.route('/<report_id>/console-log', methods=['GET'])
def get_console_log(report_id: str):
    """
    Get the console output log of the Report Agent
    
    Retrieve console output (INFO, WARNING, etc.) from the report generation process
    in real time. Unlike the agent-log endpoint which returns structured JSON logs,
    this returns plain-text console-style log lines.
    
    Query parameters:
        from_line: Line number to start reading from (optional, default 0, for incremental retrieval)
    
    Returns:
        {
            "success": true,
            "data": {
                "logs": [
                    "[19:46:14] INFO: Search complete: found 15 relevant facts",
                    "[19:46:14] INFO: Graph search: graph_id=xxx, query=...",
                    ...
                ],
                "total_lines": 100,
                "from_line": 0,
                "has_more": false
            }
        }
    """
    try:
        from_line = request.args.get('from_line', 0, type=int)
        
        log_data = ReportManager.get_console_log(report_id, from_line=from_line)
        
        return jsonify({
            "success": True,
            "data": log_data
        })
        
    except Exception as e:
        logger.error(f"Failed to get console log: {str(e)}")
        return jsonify(error_response(str(e), 500, e)), 500


@report_bp.route('/<report_id>/console-log/stream', methods=['GET'])
def stream_console_log(report_id: str):
    """
    Get the complete console log (all at once)
    
    Returns:
        {
            "success": true,
            "data": {
                "logs": [...],
                "count": 100
            }
        }
    """
    try:
        logs = ReportManager.get_console_log_stream(report_id)
        
        return jsonify({
            "success": True,
            "data": {
                "logs": logs,
                "count": len(logs)
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get console log: {str(e)}")
        return jsonify(error_response(str(e), 500, e)), 500


# ============== Tool call endpoints (for debugging) ==============

@report_bp.route('/tools/search', methods=['POST'])
def search_graph_tool():
    """
    Graph search tool endpoint (for debugging)
    
    Request (JSON):
        {
            "graph_id": "mirofish_xxxx",
            "query": "search query",
            "limit": 10
        }
    """
    try:
        data = request.get_json() or {}
        
        graph_id = data.get('graph_id')
        query = data.get('query')
        limit = data.get('limit', 10)
        
        if not graph_id or not query:
            return jsonify({
                "success": False,
                "error": "Please provide graph_id and query"
            }), 400
        
        from ..services.zep_tools import ZepToolsService
        
        tools = ZepToolsService()
        result = tools.search_graph(
            graph_id=graph_id,
            query=query,
            limit=limit
        )
        
        return jsonify({
            "success": True,
            "data": result.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Graph search failed: {str(e)}")
        return jsonify(error_response(str(e), 500, e)), 500


@report_bp.route('/tools/statistics', methods=['POST'])
def get_graph_statistics_tool():
    """
    Graph statistics tool endpoint (for debugging)
    
    Request (JSON):
        {
            "graph_id": "mirofish_xxxx"
        }
    """
    try:
        data = request.get_json() or {}
        
        graph_id = data.get('graph_id')
        
        if not graph_id:
            return jsonify({
                "success": False,
                "error": "Please provide graph_id"
            }), 400
        
        from ..services.zep_tools import ZepToolsService
        
        tools = ZepToolsService()
        result = tools.get_graph_statistics(graph_id)
        
        return jsonify({
            "success": True,
            "data": result
        })
        
    except Exception as e:
        logger.error(f"Failed to get graph statistics: {str(e)}")
        return jsonify(error_response(str(e), 500, e)), 500
