"""Caller-supplied chunk ids in VectorRAG (metadata["doc_id"]).

Default chunk ids hash only owner+text, and both write paths SKIP a chunk
whose id already exists. Two project files with byte-identical content (same
composite owner key) would therefore collide: the second file's chunks get
silently dropped and the surviving chunk carries only the first file's
source metadata — making per-file delete_by_source() unsound. The override
lets callers mint per-source ids; without it the legacy behavior is pinned
unchanged so existing personal-docs ids never churn.
"""
from src.rag_vector import _generate_doc_id

from tests.helpers.fake_vector import make_vector_rag


def test_add_document_honors_caller_doc_id():
    rag, col = make_vector_rag()
    assert rag.add_document("hello world", {"owner": "alice", "doc_id": "doc_custom1"})
    assert "doc_custom1" in col.store


def test_add_document_falls_back_to_legacy_id():
    rag, col = make_vector_rag()
    assert rag.add_document("hello world", {"owner": "alice"})
    assert _generate_doc_id("hello world", "alice") in col.store


def test_batch_identical_text_distinct_doc_ids_both_stored():
    rag, col = make_vector_rag()
    text = "identical chunk content"
    res = rag.add_documents_batch([
        (text, {"owner": "alice|project:p1", "source": "project:p1:f1", "doc_id": "doc_f1c0"}),
        (text, {"owner": "alice|project:p1", "source": "project:p1:f2", "doc_id": "doc_f2c0"}),
    ])
    assert res["success"]
    assert {"doc_f1c0", "doc_f2c0"} <= set(col.store)
    assert col.store["doc_f1c0"]["metadata"]["source"] == "project:p1:f1"
    assert col.store["doc_f2c0"]["metadata"]["source"] == "project:p1:f2"


def test_batch_without_doc_id_keeps_legacy_dedup():
    """Pin the legacy behavior: identical text + same owner, no doc_id →
    one chunk, the second add is a silent no-op."""
    rag, col = make_vector_rag()
    text = "identical chunk content"
    rag.add_documents_batch([(text, {"owner": "alice", "source": "a"})])
    rag.add_documents_batch([(text, {"owner": "alice", "source": "b"})])
    assert len(col.store) == 1
    (rec,) = col.store.values()
    assert rec["metadata"]["source"] == "a"


def test_delete_by_source_with_per_file_ids_removes_only_one_file():
    rag, col = make_vector_rag()
    text = "identical chunk content"
    rag.add_documents_batch([
        (text, {"owner": "k", "source": "project:p1:f1", "doc_id": "doc_f1c0"}),
        (text, {"owner": "k", "source": "project:p1:f2", "doc_id": "doc_f2c0"}),
    ])
    removed = rag.delete_by_source("project:p1:f1")
    assert removed == 1
    assert "doc_f1c0" not in col.store
    assert "doc_f2c0" in col.store
