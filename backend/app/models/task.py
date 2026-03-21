"""
Task status management
======================

Thread-safe in-memory registry for tracking the progress of long-running
background operations (graph building, simulation preparation, report
generation).

Design
------
:class:`TaskManager` is a *singleton*: all parts of the application share a
single instance via the ``__new__`` override.  Tasks are stored in a plain
``dict`` protected by a :class:`threading.Lock`.

Task lifecycle
--------------
``PENDING → PROCESSING → COMPLETED / FAILED``

Tasks created by a background thread start in ``PENDING`` state.  The
thread immediately updates the status to ``PROCESSING`` once it begins work,
then calls either :meth:`TaskManager.complete_task` or
:meth:`TaskManager.fail_task` when done.

Memory management
-----------------
:meth:`TaskManager.cleanup_old_tasks` removes ``COMPLETED`` and ``FAILED``
tasks older than ``max_age_hours`` (default 24 h).  The Flask app should
call this periodically or via a scheduled job to prevent unbounded growth.
"""

import uuid
import threading
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


class TaskStatus(str, Enum):
    """Background task lifecycle status.

    States
    ------
    ``PENDING``     — task created but the background thread has not started yet.
    ``PROCESSING``  — thread is actively running.
    ``COMPLETED``   — thread finished successfully; ``result`` is populated.
    ``FAILED``      — thread raised an unhandled exception; ``error`` is populated.
    """
    PENDING = "pending"          # Waiting
    PROCESSING = "processing"    # Processing
    COMPLETED = "completed"      # Completed
    FAILED = "failed"            # Failed


@dataclass
class Task:
    """Mutable record for a single background operation.

    All numeric fields default to zero / empty so that a freshly created task
    can be serialised to JSON without further initialisation.

    Attributes:
        task_id: UUID string identifying this task.
        task_type: Free-form string (e.g. ``"report_generate"``).
        status: Current :class:`TaskStatus`.
        created_at / updated_at: Timezone-naive ``datetime`` objects.
        progress: 0–100 integer percentage reported by the background thread.
        message: Human-readable status message for the frontend.
        result: Arbitrary dict set by the thread on successful completion.
        error: Error message (and optional traceback) set on failure.
        metadata: Extra key/value pairs attached at creation time
            (e.g. ``{"simulation_id": "…"}``).
        progress_detail: Fine-grained sub-step progress breakdown.
    """
    task_id: str
    task_type: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    progress: int = 0              # Overall progress percentage 0-100
    message: str = ""              # Status message
    result: Optional[Dict] = None  # Task result
    error: Optional[str] = None    # Error information
    metadata: Dict = field(default_factory=dict)  # Additional metadata
    progress_detail: Dict = field(default_factory=dict)  # Detailed progress information
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "progress": self.progress,
            "message": self.message,
            "progress_detail": self.progress_detail,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
        }


class TaskManager:
    """Thread-safe singleton registry for background task progress.

    Usage::

        tm = TaskManager()              # always returns the same instance
        task_id = tm.create_task("report_generate", metadata={"report_id": rid})
        # … in background thread:
        tm.update_task(task_id, status=TaskStatus.PROCESSING, progress=10)
        tm.complete_task(task_id, result={"report_id": rid})
        # … in API handler:
        task = tm.get_task(task_id)
        return jsonify(task.to_dict())
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._tasks: Dict[str, Task] = {}
                    cls._instance._task_lock = threading.Lock()
        return cls._instance
    
    def create_task(self, task_type: str, metadata: Optional[Dict] = None) -> str:
        """
        Create a new task
        
        Args:
            task_type: Task type
            metadata: Additional metadata
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        now = datetime.now()
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
            metadata=metadata or {}
        )
        
        with self._task_lock:
            self._tasks[task_id] = task
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task"""
        with self._task_lock:
            return self._tasks.get(task_id)
    
    def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        result: Optional[Dict] = None,
        error: Optional[str] = None,
        progress_detail: Optional[Dict] = None
    ):
        """
        Update task status
        
        Args:
            task_id: Task ID
            status: New status
            progress: Progress value
            message: Status message
            result: Task result
            error: Error information
            progress_detail: Detailed progress information
        """
        with self._task_lock:
            task = self._tasks.get(task_id)
            if task:
                task.updated_at = datetime.now()
                if status is not None:
                    task.status = status
                if progress is not None:
                    task.progress = progress
                if message is not None:
                    task.message = message
                if result is not None:
                    task.result = result
                if error is not None:
                    task.error = error
                if progress_detail is not None:
                    task.progress_detail = progress_detail
    
    def complete_task(self, task_id: str, result: Dict):
        """Mark a task as completed"""
        self.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=100,
            message="Task completed",
            result=result
        )
    
    def fail_task(self, task_id: str, error: str):
        """Mark a task as failed"""
        self.update_task(
            task_id,
            status=TaskStatus.FAILED,
            message="Task failed",
            error=error
        )
    
    def list_tasks(self, task_type: Optional[str] = None) -> list:
        """List tasks"""
        with self._task_lock:
            tasks = list(self._tasks.values())
            if task_type:
                tasks = [t for t in tasks if t.task_type == task_type]
            return [t.to_dict() for t in sorted(tasks, key=lambda x: x.created_at, reverse=True)]
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Clean up old tasks"""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        with self._task_lock:
            old_ids = [
                tid for tid, task in self._tasks.items()
                if task.created_at < cutoff and task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
            ]
            for tid in old_ids:
                del self._tasks[tid]
