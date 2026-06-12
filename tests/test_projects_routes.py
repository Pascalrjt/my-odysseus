"""Projects API: CRUD, ownership 404s, file upload validation/dedup, and
session project_id wiring (create/PATCH + /api/sessions payload)."""
import io
import tempfile
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

import core.database as cdb
from core.database import Project, ProjectFile, Session as DbSession

from tests.helpers.fake_vector import FakeUploadHandler, make_vector_rag

_TMPDB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_ENGINE = create_engine(
    f"sqlite:///{_TMPDB.name}",
    connect_args={"check_same_thread": False},
    poolclass=NullPool,
)
cdb.Base.metadata.create_all(_ENGINE)
_TS = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False)


def _route(router, path, method):
    # Last match: session_routes uses a module-level router, so each
    # setup_session_routes() call appends a fresh set of routes — the last
    # one is bound to THIS test's dependencies.
    found = None
    for r in router.routes:
        if r.path == path and method in getattr(r, "methods", set()):
            found = r.endpoint
    if found is None:
        raise AssertionError(f"route not found: {method} {path}")
    return found


def _wipe():
    db = _TS()
    try:
        db.query(ProjectFile).delete()
        db.query(DbSession).delete()
        db.query(Project).delete()
        db.commit()
    finally:
        db.close()


@pytest.fixture
def projects_api(monkeypatch, tmp_path):
    import routes.projects_routes as pr
    import src.project_files as pfiles

    _wipe()
    monkeypatch.setattr(pr, "SessionLocal", _TS)
    monkeypatch.setattr(pr, "effective_user", lambda request: "alice")
    monkeypatch.setattr(pfiles, "PROJECT_FILES_ROOT", str(tmp_path))
    rag, col = make_vector_rag()
    session_manager = MagicMock()
    session_manager.sessions = {}
    router = pr.setup_projects_routes(session_manager, FakeUploadHandler(), rag)
    return SimpleNamespace(router=router, rag=rag, col=col,
                           session_manager=session_manager, pr=pr)


def _seed_project(owner="alice", name="Research", system_prompt="Be terse."):
    db = _TS()
    try:
        p = Project(id=uuid.uuid4().hex, name=name, description="",
                    system_prompt=system_prompt, owner=owner)
        db.add(p)
        db.commit()
        return p.id
    finally:
        db.close()


def _upload(filename, content):
    return SimpleNamespace(filename=filename, file=io.BytesIO(content))


def test_create_and_list_projects(projects_api):
    create = _route(projects_api.router, "/api/projects", "POST")
    out = create(request=None, name="My Project", description="d", system_prompt="sp")
    assert out["name"] == "My Project" and out["system_prompt"] == "sp"
    assert out["file_count"] == 0 and out["chat_count"] == 0

    listing = _route(projects_api.router, "/api/projects", "GET")(request=None)
    assert [p["name"] for p in listing] == ["My Project"]


def test_create_requires_name(projects_api):
    from fastapi import HTTPException
    create = _route(projects_api.router, "/api/projects", "POST")
    with pytest.raises(HTTPException) as exc:
        create(request=None, name="   ", description="", system_prompt="")
    assert exc.value.status_code == 400


def test_foreign_project_404s_everywhere(projects_api):
    from fastapi import HTTPException
    pid = _seed_project(owner="bob")
    for path, method, kwargs in [
        ("/api/projects/{pid}", "GET", {}),
        ("/api/projects/{pid}", "PATCH", {"name": "x", "description": None, "system_prompt": None}),
        ("/api/projects/{pid}", "DELETE", {}),
        ("/api/projects/{pid}/reindex", "POST", {}),
    ]:
        endpoint = _route(projects_api.router, path, method)
        with pytest.raises(HTTPException) as exc:
            endpoint(request=None, pid=pid, **kwargs)
        assert exc.value.status_code == 404, f"{method} {path}"


def test_patch_updates_fields(projects_api):
    pid = _seed_project()
    patch = _route(projects_api.router, "/api/projects/{pid}", "PATCH")
    out = patch(request=None, pid=pid, name="Renamed", description=None,
                system_prompt="New prompt")
    assert out["name"] == "Renamed"
    assert out["system_prompt"] == "New prompt"


def test_file_upload_extracts_and_indexes(projects_api):
    pid = _seed_project()
    upload = _route(projects_api.router, "/api/projects/{pid}/files", "POST")
    body = ("project notes line\n" * 50).encode()
    out = upload(request=None, pid=pid, file=_upload("notes.md", body))
    assert out["extract_status"] == "ok"
    assert out["extracted_chars"] > 0
    assert out["indexed"] is True
    assert out["is_duplicate"] is False
    # Chunks landed in the vector store under the composite owner key.
    metas = [r["metadata"] for r in projects_api.col.store.values()]
    assert metas and all(m["owner"] == f"alice|project:{pid}" for m in metas)
    assert all(m["source"] == f"project:{pid}:{out['id']}" for m in metas)


def test_file_upload_dedups_within_project(projects_api):
    pid = _seed_project()
    upload = _route(projects_api.router, "/api/projects/{pid}/files", "POST")
    body = b"same bytes"
    first = upload(request=None, pid=pid, file=_upload("a.txt", body))
    second = upload(request=None, pid=pid, file=_upload("b.txt", body))
    assert second["is_duplicate"] is True
    assert second["id"] == first["id"]


def test_file_upload_rejects_unsafe_and_oversize(projects_api):
    from fastapi import HTTPException
    pid = _seed_project()
    upload = _route(projects_api.router, "/api/projects/{pid}/files", "POST")
    with pytest.raises(HTTPException) as exc:
        upload(request=None, pid=pid, file=_upload("evil.exe", b"MZ"))
    assert exc.value.status_code == 400

    big = b"x" * (FakeUploadHandler.max_upload_size + 1)
    with pytest.raises(HTTPException) as exc:
        upload(request=None, pid=pid, file=_upload("big.txt", big))
    assert exc.value.status_code == 400


def test_get_project_includes_files_and_chats(projects_api):
    pid = _seed_project()
    db = _TS()
    try:
        db.add(DbSession(id="s1", name="chat one", endpoint_url="http://x",
                         model="m", owner="alice", project_id=pid))
        db.commit()
    finally:
        db.close()
    get = _route(projects_api.router, "/api/projects/{pid}", "GET")
    out = get(request=None, pid=pid)
    assert [c["id"] for c in out["chats"]] == ["s1"]
    assert out["files"] == []


# ---------------------------------------------------------------------------
# Session-side wiring
# ---------------------------------------------------------------------------

@pytest.fixture
def session_patch(monkeypatch):
    import routes.session_routes as sr
    _wipe()
    monkeypatch.setattr(sr, "SessionLocal", _TS)
    monkeypatch.setattr(sr, "effective_user", lambda request: "alice")
    monkeypatch.setattr(sr, "_verify_session_owner", lambda *a, **k: None)
    session = SimpleNamespace(project_id=None)
    session_manager = MagicMock()
    session_manager.get_session.return_value = session
    router = sr.setup_session_routes(session_manager, {})
    return SimpleNamespace(
        endpoint=_route(router, "/api/session/{sid}", "PATCH"),
        session=session, sr=sr,
    )


def _patch_kwargs(**over):
    base = dict(name=None, folder=None, model=None, endpoint_url=None,
                endpoint_id=None, project_id=None)
    base.update(over)
    return base


def test_session_patch_moves_into_project(session_patch):
    pid = _seed_project()
    db = _TS()
    try:
        db.add(DbSession(id="s1", name="c", endpoint_url="http://x", model="m", owner="alice"))
        db.commit()
    finally:
        db.close()

    out = session_patch.endpoint(request=None, sid="s1", **_patch_kwargs(project_id=pid))
    assert out["project_id"] == pid
    # DB row AND cached object updated — the chat path reads the cache.
    assert session_patch.session.project_id == pid
    db = _TS()
    try:
        assert db.query(DbSession).get("s1").project_id == pid
    finally:
        db.close()

    # "__none__" clears membership (the frontend sentinel — FastAPI drops
    # empty multipart fields to the Form(None) default, so "" can't arrive).
    out = session_patch.endpoint(request=None, sid="s1", **_patch_kwargs(project_id="__none__"))
    assert out["project_id"] is None
    assert session_patch.session.project_id is None
    db = _TS()
    try:
        assert db.query(DbSession).get("s1").project_id is None
    finally:
        db.close()


def test_session_patch_rejects_foreign_project(session_patch):
    from fastapi import HTTPException
    pid = _seed_project(owner="bob")
    db = _TS()
    try:
        db.add(DbSession(id="s1", name="c", endpoint_url="http://x", model="m", owner="alice"))
        db.commit()
    finally:
        db.close()
    with pytest.raises(HTTPException) as exc:
        session_patch.endpoint(request=None, sid="s1", **_patch_kwargs(project_id=pid))
    assert exc.value.status_code == 404


def test_sessions_list_payload_includes_project_id(monkeypatch):
    import routes.session_routes as sr
    _wipe()
    monkeypatch.setattr(sr, "SessionLocal", _TS)
    monkeypatch.setattr(sr, "effective_user", lambda request: "alice")
    pid = _seed_project()
    db = _TS()
    try:
        db.add(DbSession(id="s1", name="in project", endpoint_url="http://x",
                         model="m", owner="alice", project_id=pid))
        db.add(DbSession(id="s2", name="loose", endpoint_url="http://x",
                         model="m", owner="alice"))
        db.commit()
    finally:
        db.close()

    mem = {
        "s1": SimpleNamespace(id="s1", name="in project", model="m", endpoint_url="http://x",
                              rag=False, archived=False),
        "s2": SimpleNamespace(id="s2", name="loose", model="m", endpoint_url="http://x",
                              rag=False, archived=False),
    }
    session_manager = MagicMock()
    session_manager.get_sessions_for_user.return_value = mem
    router = sr.setup_session_routes(session_manager, {})
    listing = _route(router, "/api/sessions", "GET")(request=None)
    by_id = {s["id"]: s for s in listing}
    assert by_id["s1"]["project_id"] == pid
    assert by_id["s2"]["project_id"] is None
