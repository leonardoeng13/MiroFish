"""
Data models module
==================

In-memory and file-backed data models for the MiroFish backend.

Exported symbols
----------------
- :class:`~app.models.task.TaskManager`   — thread-safe task registry (progress tracking)
- :class:`~app.models.task.TaskStatus`    — ``PENDING / PROCESSING / COMPLETED / FAILED``
- :class:`~app.models.project.Project`    — project dataclass (files, ontology, graph info)
- :class:`~app.models.project.ProjectStatus` — ``CREATED / GRAPH_BUILDING / GRAPH_COMPLETED / …``
- :class:`~app.models.project.ProjectManager` — CRUD helpers that persist to JSON on disk

All persistence is done via plain JSON files stored under
``backend/uploads/projects/<project_id>/project.json``.  No database is
required to run the server.
"""

from .task import TaskManager, TaskStatus
from .project import Project, ProjectStatus, ProjectManager

__all__ = ['TaskManager', 'TaskStatus', 'Project', 'ProjectStatus', 'ProjectManager']
