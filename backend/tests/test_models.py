"""
Unit tests for app.models.task and app.models.project
"""

import os
import json
import pytest
from datetime import datetime, timedelta

from app.models.task import Task, TaskManager, TaskStatus
from app.models.project import Project, ProjectManager, ProjectStatus


# ===========================================================================
# TaskManager
# ===========================================================================

@pytest.fixture(autouse=True)
def fresh_task_manager():
    """Reset the TaskManager singleton between tests."""
    mgr = TaskManager()
    with mgr._task_lock:
        mgr._tasks.clear()
    yield mgr
    with mgr._task_lock:
        mgr._tasks.clear()


class TestTaskManager:
    def test_create_task_returns_string_id(self, fresh_task_manager):
        task_id = fresh_task_manager.create_task("test_type")
        assert isinstance(task_id, str)
        assert len(task_id) > 0

    def test_create_task_initial_status_is_pending(self, fresh_task_manager):
        task_id = fresh_task_manager.create_task("test_type")
        task = fresh_task_manager.get_task(task_id)
        assert task.status == TaskStatus.PENDING

    def test_create_task_with_metadata(self, fresh_task_manager):
        meta = {"key": "value", "num": 42}
        task_id = fresh_task_manager.create_task("meta_type", metadata=meta)
        task = fresh_task_manager.get_task(task_id)
        assert task.metadata == meta

    def test_get_nonexistent_task_returns_none(self, fresh_task_manager):
        assert fresh_task_manager.get_task("no-such-id") is None

    def test_update_task_status(self, fresh_task_manager):
        task_id = fresh_task_manager.create_task("t")
        fresh_task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=50)
        task = fresh_task_manager.get_task(task_id)
        assert task.status == TaskStatus.PROCESSING
        assert task.progress == 50

    def test_complete_task(self, fresh_task_manager):
        task_id = fresh_task_manager.create_task("t")
        fresh_task_manager.complete_task(task_id, result={"output": "done"})
        task = fresh_task_manager.get_task(task_id)
        assert task.status == TaskStatus.COMPLETED
        assert task.progress == 100
        assert task.result == {"output": "done"}

    def test_fail_task(self, fresh_task_manager):
        task_id = fresh_task_manager.create_task("t")
        fresh_task_manager.fail_task(task_id, error="Something went wrong")
        task = fresh_task_manager.get_task(task_id)
        assert task.status == TaskStatus.FAILED
        assert "Something went wrong" in task.error

    def test_list_tasks_returns_all(self, fresh_task_manager):
        for i in range(5):
            fresh_task_manager.create_task(f"type_{i}")
        tasks = fresh_task_manager.list_tasks()
        assert len(tasks) == 5

    def test_list_tasks_filtered_by_type(self, fresh_task_manager):
        fresh_task_manager.create_task("alpha")
        fresh_task_manager.create_task("alpha")
        fresh_task_manager.create_task("beta")
        alpha_tasks = fresh_task_manager.list_tasks(task_type="alpha")
        assert len(alpha_tasks) == 2

    def test_cleanup_old_tasks_removes_stale(self, fresh_task_manager):
        task_id = fresh_task_manager.create_task("old")
        fresh_task_manager.complete_task(task_id, result={})
        # Backdate the task to > 24 hours ago
        task = fresh_task_manager.get_task(task_id)
        task.created_at = datetime.now() - timedelta(hours=25)
        fresh_task_manager.cleanup_old_tasks(max_age_hours=24)
        assert fresh_task_manager.get_task(task_id) is None

    def test_cleanup_preserves_recent_tasks(self, fresh_task_manager):
        task_id = fresh_task_manager.create_task("recent")
        fresh_task_manager.complete_task(task_id, result={})
        fresh_task_manager.cleanup_old_tasks(max_age_hours=24)
        assert fresh_task_manager.get_task(task_id) is not None

    def test_task_to_dict_contains_required_fields(self, fresh_task_manager):
        task_id = fresh_task_manager.create_task("t", metadata={"x": 1})
        task = fresh_task_manager.get_task(task_id)
        d = task.to_dict()
        for field in ("task_id", "task_type", "status", "created_at", "updated_at",
                      "progress", "message", "result", "error", "metadata"):
            assert field in d

    def test_singleton_pattern(self):
        m1 = TaskManager()
        m2 = TaskManager()
        assert m1 is m2


# ===========================================================================
# ProjectManager
# ===========================================================================

@pytest.fixture
def isolated_project_manager(tmp_dir, monkeypatch):
    """Redirect ProjectManager storage to a temp directory."""
    monkeypatch.setattr(ProjectManager, "PROJECTS_DIR", os.path.join(tmp_dir, "projects"))
    return ProjectManager


class TestProjectManager:
    def test_create_project_returns_project(self, isolated_project_manager):
        project = isolated_project_manager.create_project(name="Test Project")
        assert project.project_id.startswith("proj_")
        assert project.name == "Test Project"
        assert project.status == ProjectStatus.CREATED

    def test_get_project_by_id(self, isolated_project_manager):
        project = isolated_project_manager.create_project("P1")
        fetched = isolated_project_manager.get_project(project.project_id)
        assert fetched is not None
        assert fetched.project_id == project.project_id

    def test_get_nonexistent_project_returns_none(self, isolated_project_manager):
        assert isolated_project_manager.get_project("proj_doesnotexist") is None

    def test_list_projects_returns_all(self, isolated_project_manager):
        isolated_project_manager.create_project("A")
        isolated_project_manager.create_project("B")
        isolated_project_manager.create_project("C")
        projects = isolated_project_manager.list_projects()
        assert len(projects) == 3

    def test_list_projects_sorted_by_created_at_desc(self, isolated_project_manager):
        for name in ["First", "Second", "Third"]:
            isolated_project_manager.create_project(name)
        projects = isolated_project_manager.list_projects()
        dates = [p.created_at for p in projects]
        assert dates == sorted(dates, reverse=True)

    def test_delete_project_removes_it(self, isolated_project_manager):
        project = isolated_project_manager.create_project("ToDelete")
        result = isolated_project_manager.delete_project(project.project_id)
        assert result is True
        assert isolated_project_manager.get_project(project.project_id) is None

    def test_delete_nonexistent_project_returns_false(self, isolated_project_manager):
        assert isolated_project_manager.delete_project("proj_ghost") is False

    def test_save_and_get_extracted_text(self, isolated_project_manager):
        project = isolated_project_manager.create_project("TextTest")
        text = "Hello, this is extracted text."
        isolated_project_manager.save_extracted_text(project.project_id, text)
        result = isolated_project_manager.get_extracted_text(project.project_id)
        assert result == text

    def test_get_extracted_text_missing_returns_none(self, isolated_project_manager):
        project = isolated_project_manager.create_project("NoText")
        assert isolated_project_manager.get_extracted_text(project.project_id) is None

    def test_save_file_to_project(self, isolated_project_manager, tmp_dir):
        project = isolated_project_manager.create_project("FileTest")
        # Create a mock file-like object
        content = b"test content"
        src = os.path.join(tmp_dir, "upload.txt")
        with open(src, "wb") as f:
            f.write(content)

        class FakeFileStorage:
            filename = "upload.txt"
            def save(self, path):
                with open(src, "rb") as src_f:
                    with open(path, "wb") as dst_f:
                        dst_f.write(src_f.read())

        info = isolated_project_manager.save_file_to_project(
            project.project_id, FakeFileStorage(), "upload.txt"
        )
        assert info["original_filename"] == "upload.txt"
        assert info["size"] == len(content)
        assert os.path.exists(info["path"])

    def test_project_status_transitions(self, isolated_project_manager):
        project = isolated_project_manager.create_project("Status")
        assert project.status == ProjectStatus.CREATED
        project.status = ProjectStatus.ONTOLOGY_GENERATED
        isolated_project_manager.save_project(project)
        reloaded = isolated_project_manager.get_project(project.project_id)
        assert reloaded.status == ProjectStatus.ONTOLOGY_GENERATED

    def test_project_to_dict_and_from_dict_roundtrip(self, isolated_project_manager):
        project = isolated_project_manager.create_project("Roundtrip")
        project.graph_id = "graph_abc"
        project.simulation_requirement = "Predict outcome"
        d = project.to_dict()
        restored = Project.from_dict(d)
        assert restored.project_id == project.project_id
        assert restored.graph_id == project.graph_id
        assert restored.simulation_requirement == project.simulation_requirement
