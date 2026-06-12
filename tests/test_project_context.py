"""Project context injection: hybrid inline/RAG decision, trust-zone
placement inside build_context_preface, and owner resolution."""
import tempfile
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

import core.database as cdb
from core.database import Project, ProjectFile

from src.project_context import (
    PROJECT_INLINE_BUDGET,
    build_project_context,
)
from src.prompt_security import UNTRUSTED_CONTEXT_POLICY

_TMPDB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_ENGINE = create_engine(
    f"sqlite:///{_TMPDB.name}",
    connect_args={"check_same_thread": False},
    poolclass=NullPool,
)
cdb.Base.metadata.create_all(_ENGINE)
_TS = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False)


@pytest.fixture(autouse=True)
def _db(monkeypatch):
    """Point the module under test at the temp DB and start clean."""
    monkeypatch.setattr(cdb, "SessionLocal", _TS)
    db = _TS()
    try:
        db.query(ProjectFile).delete()
        db.query(Project).delete()
        db.commit()
    finally:
        db.close()


def _seed(owner="alice", system_prompt="Be terse.", files=()):
    db = _TS()
    try:
        pid = uuid.uuid4().hex
        db.add(Project(id=pid, name="Research", system_prompt=system_prompt, owner=owner))
        for filename, text in files:
            db.add(ProjectFile(
                id=uuid.uuid4().hex, project_id=pid, owner=owner,
                filename=filename, stored_path=f"/tmp/{filename}",
                extracted_text=text, extracted_chars=len(text),
                extract_status="ok" if text else "empty",
            ))
        db.commit()
        return pid
    finally:
        db.close()


class FakeRag:
    healthy = True

    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def search(self, query, k=5, owner=None):
        self.calls.append({"query": query, "k": k, "owner": owner})
        return self.results


def test_missing_project_returns_empty():
    sys_msg, ctx, sources = build_project_context("nope", "alice", "hi")
    assert sys_msg is None and ctx == [] and sources == []


def test_foreign_project_returns_empty():
    pid = _seed(owner="bob")
    sys_msg, ctx, sources = build_project_context(pid, "alice", "hi")
    assert sys_msg is None and ctx == []


def test_prompt_only_project():
    pid = _seed(files=())
    sys_msg, ctx, _ = build_project_context(pid, "alice", "hi")
    assert sys_msg == {"role": "system", "content": "Be terse."}
    assert ctx == []  # no files → no listing


def test_inline_mode_at_budget_boundary():
    text = "x" * PROJECT_INLINE_BUDGET
    pid = _seed(files=[("doc.txt", text)])
    rag = FakeRag()
    sys_msg, ctx, _ = build_project_context(pid, "alice", "hi", rag)
    # Exactly at budget → still inline, no retrieval.
    assert rag.calls == []
    assert len(ctx) == 2  # listing + full contents
    assert "doc.txt" in ctx[0]["content"]
    assert text in ctx[1]["content"]
    # File-derived blocks are untrusted-wrapped, never system role.
    assert all(m["role"] != "system" for m in ctx)


def test_rag_mode_over_budget():
    text = "x" * (PROJECT_INLINE_BUDGET + 1)
    pid = _seed(files=[("big.txt", text)])
    rag = FakeRag(results=[
        {"document": "relevant chunk", "similarity": 0.9,
         "metadata": {"filename": "big.txt"}},
        {"document": "noise", "similarity": 0.1,
         "metadata": {"filename": "big.txt"}},
    ])
    sys_msg, ctx, sources = build_project_context(pid, "alice", "what about x?", rag)
    # Retrieval scoped to the composite owner key.
    assert rag.calls[0]["owner"] == f"alice|project:{pid}"
    # Listing still present; excerpts filtered by threshold.
    assert len(ctx) == 2
    assert "relevant chunk" in ctx[1]["content"]
    assert "noise" not in ctx[1]["content"]
    assert sources == [{"filename": "big.txt", "snippet": "relevant chunk",
                        "similarity": 0.9}]


def test_rag_down_degrades_to_truncated_inline():
    text = "y" * (PROJECT_INLINE_BUDGET + 1)
    pid = _seed(files=[("big.txt", text)])
    sys_msg, ctx, _ = build_project_context(pid, "alice", "hi", rag_manager=None)
    assert len(ctx) == 2
    assert "search is currently unavailable" in ctx[1]["content"]
    assert text not in ctx[1]["content"]  # full text NOT inlined — truncated


def test_unsupported_files_listed_by_name_only():
    pid = _seed(files=[("photo.png", "")])
    _, ctx, _ = build_project_context(pid, "alice", "hi")
    assert len(ctx) == 1  # listing only, no content block
    assert "photo.png" in ctx[0]["content"]
    assert "no extractable text" in ctx[0]["content"]


# ---------------------------------------------------------------------------
# Wiring inside build_context_preface
# ---------------------------------------------------------------------------

def _build_preface(session, owner="alice", preset="PRESET PROMPT"):
    from src.chat_processor import ChatProcessor
    cp = ChatProcessor(MagicMock(), SimpleNamespace(rag_manager=None))
    preface, rag_sources, web_sources = cp.build_context_preface(
        message="hello there",
        session=session,
        use_web=False,
        use_rag=False,
        use_memory=False,
        preset_system_prompt=preset,
        owner=owner,
    )
    return preface


def test_preface_trust_zone_ordering():
    pid = _seed(files=[("notes.md", "project knowledge here")])
    session = SimpleNamespace(owner="alice", project_id=pid)
    preface = _build_preface(session)

    contents = [m["content"] for m in preface]
    i_preset = contents.index("PRESET PROMPT")
    i_project_prompt = contents.index("Be terse.")
    i_policy = contents.index(UNTRUSTED_CONTEXT_POLICY)
    i_files = next(i for i, c in enumerate(contents) if "project knowledge here" in c)

    # Trusted prompts stack before the policy; file content comes after it.
    assert i_preset < i_project_prompt < i_policy < i_files
    assert preface[i_project_prompt]["role"] == "system"
    assert preface[i_files]["role"] != "system"


def test_preface_uses_session_owner_not_request_user():
    """Bearer-token calls resolve get_current_user() to "api" while the
    session belongs to the token owner — project context must still inject."""
    pid = _seed(owner="alice", files=[("notes.md", "project knowledge here")])
    session = SimpleNamespace(owner="alice", project_id=pid)
    preface = _build_preface(session, owner="api")
    assert any("project knowledge here" in m["content"] for m in preface)
    assert any(m["content"] == "Be terse." for m in preface)


def test_preface_without_project_is_unchanged():
    session = SimpleNamespace(owner="alice", project_id=None)
    preface = _build_preface(session)
    assert not any("Be terse." in m["content"] for m in preface)


def test_preface_survives_deleted_project():
    """project_id pointing at a deleted row → normal chat, no crash."""
    session = SimpleNamespace(owner="alice", project_id="deleted-project-id")
    preface = _build_preface(session)
    assert any(m["content"] == "PRESET PROMPT" for m in preface)
