"""Shared fakes for project/vector tests.

FakeCollection mimics the slice of the ChromaDB collection API that
VectorRAG uses (get by ids / where-equality, add, delete), so tests can run
the REAL VectorRAG add/delete logic — including the id-dedup behavior the
project-file doc_id scheme exists to work around — without a Chroma server.
"""
import hashlib
from types import SimpleNamespace


class FakeCollection:
    def __init__(self, name="fake"):
        self.name = name
        self.store = {}  # id -> {"document": str, "metadata": dict}

    def get(self, ids=None, where=None, include=None):
        if ids is not None:
            found = [i for i in ids if i in self.store]
        elif where is not None:
            found = [
                i for i, rec in self.store.items()
                if all(rec["metadata"].get(k) == v for k, v in where.items())
            ]
        else:
            found = list(self.store)
        return {
            "ids": found,
            "documents": [self.store[i]["document"] for i in found],
            "metadatas": [self.store[i]["metadata"] for i in found],
        }

    def add(self, ids, embeddings=None, documents=None, metadatas=None):
        for i, d, m in zip(ids, documents, metadatas):
            self.store[i] = {"document": d, "metadata": m}

    def delete(self, ids):
        for i in ids:
            self.store.pop(i, None)


def make_vector_rag():
    """A real VectorRAG instance wired to one fake lane/collection."""
    from src.rag_vector import VectorRAG
    rag = object.__new__(VectorRAG)
    rag._healthy = True
    rag._collection = None
    rag._model = None
    col = FakeCollection()
    rag._lanes = [SimpleNamespace(
        name="fake",
        collection=col,
        encode=lambda texts: [[0.0] for _ in texts],
    )]
    return rag, col


class FakeUploadHandler:
    """Just the validators save_project_file borrows from UploadHandler."""
    max_upload_size = 10 * 1024 * 1024

    def detect_content_type(self, file_obj, original_filename):
        return "text/plain"

    def is_safe_file_type(self, content_type, filename):
        return not filename.lower().endswith((".exe", ".dll", ".bat"))

    def calculate_file_hash(self, file_obj):
        file_obj.seek(0)
        digest = hashlib.sha256(file_obj.read()).hexdigest()
        file_obj.seek(0)
        return digest
