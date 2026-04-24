"""
Graph-related API routes
========================

Implements the two-step knowledge-graph pipeline:

Step 1 — ``POST /api/graph/ontology/generate``
    Accept one or more document files (PDF / Markdown / TXT) plus a plain-text
    *simulation requirement*, extract the text, then ask the LLM to produce an
    entity/relationship ontology.  A :class:`~app.models.project.Project` is
    persisted on disk so subsequent calls can reference it by ``project_id``.

Step 2 — ``POST /api/graph/build``
    Read the previously extracted text and ontology from disk, split it into
    overlapping chunks, and stream each chunk into a Zep knowledge graph via
    :class:`~app.services.graph_builder.GraphBuilderService`.  The build runs
    in a background daemon thread; progress is tracked via
    :class:`~app.models.task.TaskManager` and polled via
    ``GET /api/graph/task/<task_id>``.

Additional endpoints
--------------------
- ``GET    /api/graph/project/<project_id>``       — retrieve project metadata
- ``GET    /api/graph/project/list``               — list all projects
- ``DELETE /api/graph/project/<project_id>``       — remove project directory
- ``POST   /api/graph/project/<project_id>/reset`` — revert project to ONTOLOGY_GENERATED
- ``GET    /api/graph/task/<task_id>``             — poll background task progress
- ``GET    /api/graph/tasks``                      — list all tasks
- ``GET    /api/graph/data/<graph_id>``            — raw node/edge data from Zep
- ``DELETE /api/graph/delete/<graph_id>``          — remove a Zep graph

Server-side state model
-----------------------
All project state is stored as JSON under
``backend/uploads/projects/<project_id>/``.  No database is required.
The frontend only needs to pass the ``project_id`` between API 1 and API 2.
"""

import os
import traceback
import threading
from flask import request, jsonify

from . import graph_bp
from ..config import Config
from ..services.ontology_generator import OntologyGenerator
from ..services.graph_builder import GraphBuilderService
from ..services.text_processor import TextProcessor
from ..services.location_aggregator import LocationAggregatorService
from ..utils.file_parser import FileParser
from ..utils.logger import get_logger
from ..utils.response import error_response
from ..utils.validators import BuildGraphRequest, parse_request
from ..models.task import TaskManager, TaskStatus
from ..models.project import ProjectManager, ProjectStatus

# Get logger
logger = get_logger('mirofish.api')


def allowed_file(filename: str) -> bool:
    """Check whether the file extension is in the server's allow-list.

    Args:
        filename: The original filename supplied by the client.  Can be
            ``None`` or an empty string, in which case ``False`` is returned.

    Returns:
        ``True`` when the extension (lower-cased, without the leading dot)
        is present in :attr:`~app.config.Config.ALLOWED_EXTENSIONS`
        (``{'pdf', 'md', 'txt', 'markdown'}``).
    """
    if not filename or '.' not in filename:
        return False
    ext = os.path.splitext(filename)[1].lower().lstrip('.')
    return ext in Config.ALLOWED_EXTENSIONS


# ============== Project management endpoints ==============

@graph_bp.route('/project/<project_id>', methods=['GET'])
def get_project(project_id: str):
    """
    Get project details
    """
    project = ProjectManager.get_project(project_id)
    
    if not project:
        return jsonify({
            "success": False,
            "error": f"Project not found: {project_id}"
        }), 404
    
    return jsonify({
        "success": True,
        "data": project.to_dict()
    })


@graph_bp.route('/project/list', methods=['GET'])
def list_projects():
    """
    List all projects
    """
    limit = request.args.get('limit', 50, type=int)
    projects = ProjectManager.list_projects(limit=limit)
    
    return jsonify({
        "success": True,
        "data": [p.to_dict() for p in projects],
        "count": len(projects)
    })


@graph_bp.route('/project/<project_id>', methods=['DELETE'])
def delete_project(project_id: str):
    """
    Delete a project
    """
    success = ProjectManager.delete_project(project_id)
    
    if not success:
        return jsonify({
            "success": False,
            "error": f"Project not found or deletion failed: {project_id}"
        }), 404
    
    return jsonify({
        "success": True,
        "message": f"Project deleted: {project_id}"
    })


@graph_bp.route('/project/<project_id>/reset', methods=['POST'])
def reset_project(project_id: str):
    """
    Reset project status (used to rebuild the graph)
    """
    project = ProjectManager.get_project(project_id)
    
    if not project:
        return jsonify({
            "success": False,
            "error": f"Project not found: {project_id}"
        }), 404
    
    # Reset to ontology-generated state
    if project.ontology:
        project.status = ProjectStatus.ONTOLOGY_GENERATED
    else:
        project.status = ProjectStatus.CREATED
    
    project.graph_id = None
    project.graph_build_task_id = None
    project.error = None
    ProjectManager.save_project(project)
    
    return jsonify({
        "success": True,
        "message": f"Project reset: {project_id}",
        "data": project.to_dict()
    })


# ============== API 1: Upload files and generate ontology ==============

@graph_bp.route('/ontology/generate', methods=['POST'])
def generate_ontology():
    """
    API 1: Upload files and generate an ontology definition
    
    Request method: multipart/form-data
    
    Parameters:
        files: Uploaded files (PDF/MD/TXT), multiple allowed
        simulation_requirement: Simulation requirement description (required)
        project_name: Project name (optional)
        additional_context: Additional notes (optional)
        
    Returns:
        {
            "success": true,
            "data": {
                "project_id": "proj_xxxx",
                "ontology": {
                    "entity_types": [...],
                    "edge_types": [...],
                    "analysis_summary": "..."
                },
                "files": [...],
                "total_text_length": 12345
            }
        }
    """
    try:
        logger.info("=== Starting ontology generation ===")
        
        # Get parameters
        simulation_requirement = request.form.get('simulation_requirement', '')
        project_name = request.form.get('project_name', 'Unnamed Project')
        additional_context = request.form.get('additional_context', '')
        
        logger.debug(f"Project name: {project_name}")
        logger.debug(f"Simulation requirement: {simulation_requirement[:100]}...")
        
        if not simulation_requirement:
            return jsonify({
                "success": False,
                "error": "Please provide a simulation requirement description (simulation_requirement)"
            }), 400
        
        # Get uploaded files
        uploaded_files = request.files.getlist('files')
        if not uploaded_files or all(not f.filename for f in uploaded_files):
            return jsonify({
                "success": False,
                "error": "Please upload at least one document file"
            }), 400
        
        # Create project
        project = ProjectManager.create_project(name=project_name)
        project.simulation_requirement = simulation_requirement
        logger.info(f"Project created: {project.project_id}")
        
        # Save files and extract text
        document_texts = []
        all_text = ""
        
        for file in uploaded_files:
            if file and file.filename and allowed_file(file.filename):
                # Save file to project directory
                file_info = ProjectManager.save_file_to_project(
                    project.project_id, 
                    file, 
                    file.filename
                )
                project.files.append({
                    "filename": file_info["original_filename"],
                    "size": file_info["size"]
                })
                
                # Extract text
                text = FileParser.extract_text(file_info["path"])
                text = TextProcessor.preprocess_text(text)
                document_texts.append(text)
                all_text += f"\n\n=== {file_info['original_filename']} ===\n{text}"
        
        if not document_texts:
            ProjectManager.delete_project(project.project_id)
            return jsonify({
                "success": False,
                "error": "No documents were processed successfully; please check the file formats"
            }), 400
        
        # Save extracted text
        project.total_text_length = len(all_text)
        ProjectManager.save_extracted_text(project.project_id, all_text)
        logger.info(f"Text extraction complete, total {len(all_text)} characters")
        
        # Generate ontology
        logger.info("Calling LLM to generate ontology definition...")
        generator = OntologyGenerator()
        ontology = generator.generate(
            document_texts=document_texts,
            simulation_requirement=simulation_requirement,
            additional_context=additional_context if additional_context else None
        )
        
        # Save ontology to project
        entity_count = len(ontology.get("entity_types", []))
        edge_count = len(ontology.get("edge_types", []))
        logger.info(f"Ontology generated: {entity_count} entity types, {edge_count} relationship types")
        
        project.ontology = {
            "entity_types": ontology.get("entity_types", []),
            "edge_types": ontology.get("edge_types", [])
        }
        project.analysis_summary = ontology.get("analysis_summary", "")
        project.status = ProjectStatus.ONTOLOGY_GENERATED
        ProjectManager.save_project(project)
        logger.info(f"=== Ontology generation complete === Project ID: {project.project_id}")
        
        return jsonify({
            "success": True,
            "data": {
                "project_id": project.project_id,
                "project_name": project.name,
                "ontology": project.ontology,
                "analysis_summary": project.analysis_summary,
                "files": project.files,
                "total_text_length": project.total_text_length
            }
        })
        
    except Exception as e:
        return jsonify(error_response(str(e), 500, e)), 500


# ============== API 2: Build graph ==============

@graph_bp.route('/build', methods=['POST'])
def build_graph():
    """
    API 2: Build a graph based on project_id
    
    Request (JSON):
        {
            "project_id": "proj_xxxx",  // required, from API 1
            "graph_name": "Graph Name", // optional
            "chunk_size": 500,          // optional, default 500
            "chunk_overlap": 50         // optional, default 50
        }
        
    Returns:
        {
            "success": true,
            "data": {
                "project_id": "proj_xxxx",
                "task_id": "task_xxxx",
                "message": "Graph build task started"
            }
        }
    """
    try:
        logger.info("=== Starting graph build ===")
        
        # Check configuration
        errors = []
        if not Config.ZEP_API_KEY:
            errors.append("ZEP_API_KEY is not configured")
        if errors:
            logger.error(f"Configuration errors: {errors}")
            return jsonify({
                "success": False,
                "error": "Configuration error: " + "; ".join(errors)
            }), 500
        
        # Parse and validate request body
        data = request.get_json() or {}
        validated, err = parse_request(BuildGraphRequest, data)
        if err:
            return jsonify({"success": False, "error": err}), 400
        
        project_id = validated.project_id
        logger.debug(f"Request parameters: project_id={project_id}")
        
        # Get project
        project = ProjectManager.get_project(project_id)
        if not project:
            return jsonify({
                "success": False,
                "error": f"Project not found: {project_id}"
            }), 404
        
        # Check project status
        force = data.get('force', False)  # Force rebuild
        
        if project.status == ProjectStatus.CREATED:
            return jsonify({
                "success": False,
                "error": "Ontology has not been generated for this project; please call /ontology/generate first"
            }), 400
        
        if project.status == ProjectStatus.GRAPH_BUILDING and not force:
            return jsonify({
                "success": False,
                "error": "Graph is already being built; please do not resubmit. Add force: true to force a rebuild.",
                "task_id": project.graph_build_task_id
            }), 400
        
        # If force rebuilding, reset status
        if force and project.status in [ProjectStatus.GRAPH_BUILDING, ProjectStatus.FAILED, ProjectStatus.GRAPH_COMPLETED]:
            project.status = ProjectStatus.ONTOLOGY_GENERATED
            project.graph_id = None
            project.graph_build_task_id = None
            project.error = None
        
        # Get configuration – prefer validated values when explicitly provided
        graph_name = data.get('graph_name', project.name or 'MiroFish Graph')
        chunk_size = validated.chunk_size if validated.chunk_size is not None else (project.chunk_size or Config.DEFAULT_CHUNK_SIZE)
        chunk_overlap = validated.chunk_overlap if validated.chunk_overlap is not None else (project.chunk_overlap or Config.DEFAULT_CHUNK_OVERLAP)
        
        # Update project configuration
        project.chunk_size = chunk_size
        project.chunk_overlap = chunk_overlap
        
        # Get extracted text
        text = ProjectManager.get_extracted_text(project_id)
        if not text:
            return jsonify({
                "success": False,
                "error": "Extracted text content not found"
            }), 400
        
        # Get ontology
        ontology = project.ontology
        if not ontology:
            return jsonify({
                "success": False,
                "error": "Ontology definition not found"
            }), 400
        
        # Create async task
        task_manager = TaskManager()
        task_id = task_manager.create_task(f"Build graph: {graph_name}")
        logger.info(f"Graph build task created: task_id={task_id}, project_id={project_id}")
        
        # Update project status
        project.status = ProjectStatus.GRAPH_BUILDING
        project.graph_build_task_id = task_id
        ProjectManager.save_project(project)
        
        # Start background task
        def build_task():
            build_logger = get_logger('mirofish.build')
            try:
                build_logger.info(f"[{task_id}] Starting graph build...")
                task_manager.update_task(
                    task_id, 
                    status=TaskStatus.PROCESSING,
                    message="Initializing graph build service..."
                )
                
                # Create graph build service
                builder = GraphBuilderService(api_key=Config.ZEP_API_KEY)
                
                # Chunk text
                task_manager.update_task(
                    task_id,
                    message="Splitting text into chunks...",
                    progress=5
                )
                chunks = TextProcessor.split_text(
                    text, 
                    chunk_size=chunk_size, 
                    overlap=chunk_overlap
                )
                total_chunks = len(chunks)
                
                # Create graph
                task_manager.update_task(
                    task_id,
                    message="Creating Zep graph...",
                    progress=10
                )
                graph_id = builder.create_graph(name=graph_name)
                
                # Update project's graph_id
                project.graph_id = graph_id
                ProjectManager.save_project(project)
                
                # Set ontology
                task_manager.update_task(
                    task_id,
                    message="Setting ontology definition...",
                    progress=15
                )
                builder.set_ontology(graph_id, ontology)
                
                # Add text (progress_callback signature: (msg, progress_ratio))
                def add_progress_callback(msg, progress_ratio):
                    progress = 15 + int(progress_ratio * 40)  # 15% - 55%
                    task_manager.update_task(
                        task_id,
                        message=msg,
                        progress=progress
                    )
                
                task_manager.update_task(
                    task_id,
                    message=f"Adding {total_chunks} text chunks...",
                    progress=15
                )
                
                episode_uuids = builder.add_text_batches(
                    graph_id, 
                    chunks,
                    batch_size=3,
                    progress_callback=add_progress_callback
                )
                
                # Wait for Zep to finish processing (query the processed status of each episode)
                task_manager.update_task(
                    task_id,
                    message="Waiting for Zep to process data...",
                    progress=55
                )
                
                def wait_progress_callback(msg, progress_ratio):
                    progress = 55 + int(progress_ratio * 35)  # 55% - 90%
                    task_manager.update_task(
                        task_id,
                        message=msg,
                        progress=progress
                    )
                
                builder._wait_for_episodes(episode_uuids, wait_progress_callback)
                
                # Get graph data
                task_manager.update_task(
                    task_id,
                    message="Retrieving graph data...",
                    progress=95
                )
                graph_data = builder.get_graph_data(graph_id)
                
                # Update project status
                project.status = ProjectStatus.GRAPH_COMPLETED
                ProjectManager.save_project(project)
                
                node_count = graph_data.get("node_count", 0)
                edge_count = graph_data.get("edge_count", 0)
                build_logger.info(f"[{task_id}] Graph build complete: graph_id={graph_id}, nodes={node_count}, edges={edge_count}")
                
                # Complete
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.COMPLETED,
                    message="Graph build completed",
                    progress=100,
                    result={
                        "project_id": project_id,
                        "graph_id": graph_id,
                        "node_count": node_count,
                        "edge_count": edge_count,
                        "chunk_count": total_chunks
                    }
                )
                
            except Exception as e:
                # Update project status to failed
                build_logger.error(f"[{task_id}] Graph build failed: {str(e)}")
                build_logger.debug(traceback.format_exc())
                
                project.status = ProjectStatus.FAILED
                project.error = str(e)
                ProjectManager.save_project(project)
                
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    message=f"Build failed: {str(e)}",
                    error=traceback.format_exc()
                )
        
        # Start background thread
        thread = threading.Thread(target=build_task, daemon=True)
        thread.start()
        
        return jsonify({
            "success": True,
            "data": {
                "project_id": project_id,
                "task_id": task_id,
                "message": "Graph build task started; use /task/{task_id} to check progress"
            }
        })
        
    except Exception as e:
        return jsonify(error_response(str(e), 500, e)), 500


# ============== Task query endpoints ==============

@graph_bp.route('/task/<task_id>', methods=['GET'])
def get_task(task_id: str):
    """
    Query task status
    """
    task = TaskManager().get_task(task_id)
    
    if not task:
        return jsonify({
            "success": False,
            "error": f"Task not found: {task_id}"
        }), 404
    
    return jsonify({
        "success": True,
        "data": task.to_dict()
    })


@graph_bp.route('/tasks', methods=['GET'])
def list_tasks():
    """
    List all tasks
    """
    tasks = TaskManager().list_tasks()
    
    return jsonify({
        "success": True,
        "data": [t.to_dict() for t in tasks],
        "count": len(tasks)
    })


# ============== Graph data endpoints ==============

@graph_bp.route('/data/<graph_id>', methods=['GET'])
def get_graph_data(graph_id: str):
    """
    Get graph data (nodes and edges)
    """
    try:
        if not Config.ZEP_API_KEY:
            return jsonify({
                "success": False,
                "error": "ZEP_API_KEY is not configured"
            }), 500
        
        builder = GraphBuilderService(api_key=Config.ZEP_API_KEY)
        graph_data = builder.get_graph_data(graph_id)
        
        return jsonify({
            "success": True,
            "data": graph_data
        })
        
    except Exception as e:
        return jsonify(error_response(str(e), 500, e)), 500


@graph_bp.route('/delete/<graph_id>', methods=['DELETE'])
def delete_graph(graph_id: str):
    """
    Delete a Zep graph
    """
    try:
        if not Config.ZEP_API_KEY:
            return jsonify({
                "success": False,
                "error": "ZEP_API_KEY is not configured"
            }), 500
        
        builder = GraphBuilderService(api_key=Config.ZEP_API_KEY)
        builder.delete_graph(graph_id)
        
        return jsonify({
            "success": True,
            "message": f"Graph deleted: {graph_id}"
        })
        
    except Exception as e:
        return jsonify(error_response(str(e), 500, e)), 500


# ============== Location filter endpoints ==============

@graph_bp.route('/<graph_id>/location-stats', methods=['GET'])
def get_location_stats(graph_id: str):
    """
    Return geographic aggregation statistics for all entities in the graph.

    Groups entities by country → state → city → neighborhood and counts
    how many entities are found at each level.

    Query parameters
    ----------------
    entity_type : str, optional
        Filter to a specific entity type (e.g. ``Person``, ``University``).
    country : str, optional
        Restrict output to a single country (case-insensitive).
    include_unknown : bool, optional
        When ``true``, include entities whose location could not be resolved
        in an "Unknown" bucket.  Default ``false``.

    Returns
    -------
    JSON with keys ``graph_id``, ``total_entities_analysed``,
    ``total_with_location``, ``filters_applied``, ``hierarchy``.
    """
    try:
        if not Config.ZEP_API_KEY:
            return jsonify({
                "success": False,
                "error": "ZEP_API_KEY is not configured"
            }), 500

        entity_type = request.args.get('entity_type') or None
        country = request.args.get('country') or None
        include_unknown = request.args.get('include_unknown', 'false').lower() == 'true'

        svc = LocationAggregatorService(api_key=Config.ZEP_API_KEY)
        stats = svc.get_location_stats(
            graph_id=graph_id,
            entity_type_filter=entity_type,
            country_filter=country,
            include_unknown=include_unknown,
        )

        return jsonify({"success": True, "data": stats})

    except Exception as e:
        logger.error(f"Failed to get location stats for graph {graph_id}: {str(e)}")
        return jsonify({"success": False, "error": "Failed to retrieve location statistics"}), 500


@graph_bp.route('/<graph_id>/location-entities', methods=['GET'])
def get_entities_by_location(graph_id: str):
    """
    Return a flat list of entities matching the requested location.

    Query parameters
    ----------------
    country : str, optional
    state : str, optional
    city : str, optional
    neighborhood : str, optional
    entity_type : str, optional

    At least one location field must be provided.

    Returns
    -------
    JSON with ``count`` and ``entities`` list.
    """
    try:
        if not Config.ZEP_API_KEY:
            return jsonify({
                "success": False,
                "error": "ZEP_API_KEY is not configured"
            }), 500

        country = request.args.get('country') or None
        state = request.args.get('state') or None
        city = request.args.get('city') or None
        neighborhood = request.args.get('neighborhood') or None
        entity_type = request.args.get('entity_type') or None

        if not any([country, state, city, neighborhood]):
            return jsonify({
                "success": False,
                "error": "At least one location filter (country, state, city, neighborhood) must be provided"
            }), 400

        svc = LocationAggregatorService(api_key=Config.ZEP_API_KEY)
        entities = svc.get_entities_by_location(
            graph_id=graph_id,
            country=country,
            state=state,
            city=city,
            neighborhood=neighborhood,
            entity_type_filter=entity_type,
        )

        return jsonify({
            "success": True,
            "data": {
                "count": len(entities),
                "entities": entities,
                "filters": {
                    "country": country,
                    "state": state,
                    "city": city,
                    "neighborhood": neighborhood,
                    "entity_type": entity_type,
                }
            }
        })

    except ValueError as e:
        # ValueError is raised by the service for invalid filter combinations.
        # Return a safe, fixed message rather than forwarding the internal exception string.
        logger.warning(f"Invalid location filter for graph {graph_id}: {str(e)}")
        return jsonify({"success": False, "error": "Invalid location filter: at least one of country, state, city, or neighborhood must be provided"}), 400
    except Exception as e:
        logger.error(f"Failed to get entities by location for graph {graph_id}: {str(e)}")
        return jsonify({"success": False, "error": "Failed to retrieve entities by location"}), 500
