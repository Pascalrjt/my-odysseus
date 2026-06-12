"""Deletion semantics for projects and project files.

The duplicate-chunk cases are the load-bearing ones: they prove that two
files with byte-identical content stay independently deletable (the whole
point of the per-file doc_id scheme), and that deleting one never removes
content the other still provides.
"""
import io
import os
import tempfile
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

import core.database as cdb
from core.database import ChatMessage, Project, ProjectFile, Session as DbSession

from tests.helpers.fake_vector import FakeUploadHandler, make_vector_rag

_TMPDB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_ENGINE = create_engine(
    f"sqlite:///{_TMPDB.name}",
    connect_args={"check_same_thread": False},
    poolclass=NullPool,
)
cdb.Base.metadata.create_all(_ENGINE)
_TS = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False)

# Long enough to chunk, identical across files.
SHARED_CONTENT = ("All work and no play makes Jack a dull boy. " * 60).encode()


@pytest.fixture
def env(monkeypatch, tmp_path):
    import routes.projects_routes as pr
    import src.project_files as pfiles

    db = _TS()
    try:
        for model in (ChatMessage, ProjectFile, DbSession, Project):
            db.query(model).delete()
        db.commit()
    finally:
        db.close()

    monkeypatch.setattr(pr, "SessionLocal", _TS)
    monkeypatch.setattr(pr, "effective_user", lambda request: "alice")
    monkeypatch.setattr(pfiles, "PROJECT_FILES_ROOT", str(tmp_path))

    rag, col = make_vector_rag()
    session_manager = MagicMock()
    session_manager.sessions = {}
    router = pr.setup_projects_routes(session_manager, FakeUploadHandler(), rag)

    def route(path, method):
        found = None
        for r in router.routes:
            if r.path == path and method in getattr(r, "methods", set()):
                found = r.endpoint
        assert found is not None, f"{method} {path}"
        return found

    return SimpleNamespace(route=route, rag=rag, col=col,
                           session_manager=session_manager, tmp=tmp_path)


def _make_project(env, name="P"):
    return env.route("/api/projects", "POST")(
        request=None, name=name, description="", system_prompt="")["id"]


def _upload(env, pid, filename, content=SHARED_CONTENT):
    return env.route("/api/projects/{pid}/files", "POST")(
        request=None, pid=pid,
        file=SimpleNamespace(filename=filename, file=io.BytesIO(content)))


def _sources_in_store(col):
    return {r["metadata"]["source"] for r in col.store.values()}


def test_duplicate_content_two_files_indexed_independently(env):
    pid = _make_project(env)
    f1 = _upload(env, pid, "a.txt")
    f2 = _upload(env, pid, "b.txt", SHARED_CONTENT + b" tail difference")
    # Different hashes → both stored; chunks for each file present.
    sources = _sources_in_store(env.col)
    assert f"project:{pid}:{f1['id']}" in sources
    assert f"project:{pid}:{f2['id']}" in sources
    # The shared leading chunks exist twice (per-file doc_ids — no dedup-drop).
    f1_chunks = [r for r in env.col.store.values()
                 if r["metadata"]["file_id"] == f1["id"]]
    f2_chunks = [r for r in env.col.store.values()
                 if r["metadata"]["file_id"] == f2["id"]]
    assert f1_chunks and f2_chunks
    shared = {c["document"] for c in f1_chunks} & {c["document"] for c in f2_chunks}
    assert shared, "expected at least one byte-identical chunk across the two files"


def test_delete_one_file_keeps_duplicate_content_of_other(env):
    pid = _make_project(env)
    f1 = _upload(env, pid, "a.txt")
    f2 = _upload(env, pid, "b.txt", SHARED_CONTENT + b" tail difference")

    env.route("/api/projects/{pid}/files/{fid}", "DELETE")(
        request=None, pid=pid, fid=f1["id"])

    sources = _sources_in_store(env.col)
    assert f"project:{pid}:{f1['id']}" not in sources
    assert f"project:{pid}:{f2['id']}" in sources
    # f2's copy of the shared content survives f1's deletion.
    f2_docs = {r["document"] for r in env.col.store.values()
               if r["metadata"]["file_id"] == f2["id"]}
    assert any("All work and no play" in d for d in f2_docs)

    # Row + binary gone, extracted text dead with the row.
    db = _TS()
    try:
        assert db.query(ProjectFile).filter_by(id=f1["id"]).first() is None
        f2_row = db.query(ProjectFile).filter_by(id=f2["id"]).first()
        assert f2_row is not None and f2_row.extracted_text
        assert not os.path.exists(os.path.join(str(env.tmp), pid, f"{f1['id']}.txt"))
        assert os.path.exists(f2_row.stored_path)
    finally:
        db.close()

    env.route("/api/projects/{pid}/files/{fid}", "DELETE")(
        request=None, pid=pid, fid=f2["id"])
    assert _sources_in_store(env.col) == set()


def test_same_content_in_two_projects_is_independent(env):
    pid_a = _make_project(env, "A")
    pid_b = _make_project(env, "B")
    fa = _upload(env, pid_a, "doc.txt")
    fb = _upload(env, pid_b, "doc.txt")
    # No cross-project dedup at the file level…
    assert fa["id"] != fb["id"] and not fb["is_duplicate"]
    # …and deleting project A leaves project B's chunks intact.
    env.route("/api/projects/{pid}", "DELETE")(request=None, pid=pid_a)
    sources = _sources_in_store(env.col)
    assert f"project:{pid_b}:{fb['id']}" in sources
    assert all(not s.startswith(f"project:{pid_a}:") for s in sources)


def test_reindex_same_file_does_not_duplicate_chunks(env):
    import src.project_files as pfiles
    pid = _make_project(env)
    f1 = _upload(env, pid, "a.txt")
    before = set(env.col.store)
    db = _TS()
    try:
        pf = db.query(ProjectFile).filter_by(id=f1["id"]).first()
        assert pfiles.index_project_file(pf, env.rag)  # second pass
    finally:
        db.close()
    assert set(env.col.store) == before  # deterministic ids → no growth


def test_project_delete_detaches_chats_and_removes_assets(env):
    pid = _make_project(env)
    f1 = _upload(env, pid, "a.txt")

    db = _TS()
    try:
        db.add(DbSession(id="s1", name="chat", endpoint_url="http://x",
                         model="m", owner="alice", project_id=pid))
        db.add(ChatMessage(id="m1", session_id="s1", role="user", content="hi"))
        db.commit()
    finally:
        db.close()
    # Cached entries — including a metadata-only one — must be detached too:
    # get_sessions_for_user() and the chat path serve from this cache.
    cached_full = SimpleNamespace(project_id=pid, history=["..."])
    cached_meta_only = SimpleNamespace(project_id=pid, history=[])
    other = SimpleNamespace(project_id="other-project", history=[])
    env.session_manager.sessions = {
        "s1": cached_full, "s2": cached_meta_only, "s3": other,
    }

    env.route("/api/projects/{pid}", "DELETE")(request=None, pid=pid)

    db = _TS()
    try:
        assert db.query(Project).filter_by(id=pid).first() is None
        assert db.query(ProjectFile).filter_by(project_id=pid).count() == 0
        # Chat survives with history, detached.
        row = db.query(DbSession).get("s1")
        assert row is not None and row.project_id is None
        assert db.query(ChatMessage).filter_by(session_id="s1").count() == 1
    finally:
        db.close()

    assert cached_full.project_id is None
    assert cached_meta_only.project_id is None
    assert other.project_id == "other-project"  # untouched
    # Binaries directory swept.
    assert not os.path.exists(os.path.join(str(env.tmp), pid))
    # Vector store cleaned.
    assert _sources_in_store(env.col) == set()
