"""Project file storage, extraction, and vector indexing.

Project files are uploaded once on the /projects page and shared by every
chat in that project. Three design constraints shape this module:

1. Binaries live under data/project_files/{project_id}/ — deliberately
   OUTSIDE data/uploads/, because UploadHandler.cleanup_old_uploads() walks
   the uploads tree and deletes whole date directories older than ~30 days
   with no per-file exclusion. A sibling directory is structurally immune,
   and whole-project deletion becomes a single rmtree.

2. Extracted text is stored on the ProjectFile row (not a sidecar file) so
   it lives and dies transactionally with the row. The chat-path processors
   in document_processor.py are NOT reused for extraction: they hard-cap at
   15-30K chars and wrap output in chat formatting, which would silently
   cripple RAG over large files. We read with the lower-level helpers and
   apply our own, much higher cap.

3. Vector chunks are indexed with a caller-supplied, per-file deterministic
   doc_id (owner_key|source|chunk_index). The default id scheme hashes only
   owner+text, so byte-identical chunks in two different files would
   collide: the second file's chunks would be silently dropped and carry
   the first file's source metadata — making delete_by_source() unsound.
   Per-file ids keep duplicate content independently deletable.
"""

import hashlib
import logging
import os
import shutil
from typing import Any, Optional, Tuple

from src.upload_handler import secure_filename

logger = logging.getLogger(__name__)

# Total extracted chars kept per file. Far above the inline budget (the
# overflow is what project RAG mode exists for), bounded so a pathological
# text file can't balloon the DB row or the embedding workload.
PROJECT_FILE_EXTRACT_CAP = 120_000

PROJECT_CHUNK_SIZE = 1000
PROJECT_CHUNK_OVERLAP = 200

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_FILES_ROOT = os.environ.get(
    "PROJECT_FILES_DIR", os.path.join(_BASE_DIR, "data", "project_files")
)

_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".log", ".json", ".yml",
    ".yaml", ".xml", ".html", ".htm", ".css", ".sql", ".sh", ".bash",
    ".nix", ".py", ".js", ".ts", ".jsx", ".tsx", ".c", ".cpp", ".h",
    ".java", ".go", ".rs", ".php", ".rb",
}
_UNSUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg",
    ".webm", ".wav", ".mp3", ".m4a", ".ogg", ".mp4", ".mov",
}


def project_owner_key(owner: Optional[str], project_id: str) -> str:
    """Composite owner key used for vector-store scoping.

    VectorRAG.search() filters on exact metadata['owner'] equality, so this
    key makes project chunks invisible to personal-docs RAG (bare username)
    and to every other project/user — isolation comes for free."""
    return f"{owner or ''}|project:{project_id}"


def project_file_source(project_id: str, file_id: str) -> str:
    """Per-file source tag; delete_by_source(source) removes one file's chunks."""
    return f"project:{project_id}:{file_id}"


def project_chunk_doc_id(owner_key: str, source: str, chunk_index: int) -> str:
    """Deterministic per-file chunk id (see module docstring, point 3).

    Deterministic so re-indexing the same file dedups against itself."""
    key = f"{owner_key}|{source}|{chunk_index}"
    return f"doc_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}"


def project_files_dir(project_id: str, create: bool = True) -> str:
    """Directory holding one project's binaries."""
    path = os.path.join(PROJECT_FILES_ROOT, project_id)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def inside_project_files(path: str) -> bool:
    """Path-confinement guard for delete/serve operations."""
    base = os.path.realpath(PROJECT_FILES_ROOT)
    p = os.path.realpath(path)
    try:
        return os.path.commonpath([base, p]) == base
    except Exception:
        return False


# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------

def _read_text_with_fallback(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            from charset_normalizer import detect
            encoding = (detect(raw) or {}).get("encoding") or "utf-8"
            return raw.decode(encoding, errors="replace")
        except Exception:
            return raw.decode("utf-8", errors="replace")


def _extract_pdf_text(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    parts = []
    total = 0
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            parts.append(text)
            total += len(text)
        if total >= PROJECT_FILE_EXTRACT_CAP:
            break
    return "\n\n".join(parts)


def extract_project_file_text(path: str, filename: str) -> Tuple[str, str]:
    """Extract plain text from a project file.

    Returns (text, status) with status one of ok | empty | failed | unsupported.
    Text is capped at PROJECT_FILE_EXTRACT_CAP with a visible marker."""
    _, ext = os.path.splitext(filename.lower())
    try:
        if ext in _UNSUPPORTED_EXTENSIONS:
            return "", "unsupported"
        if ext == ".pdf":
            text = _extract_pdf_text(path)
        elif ext in _TEXT_EXTENSIONS:
            text = _read_text_with_fallback(path)
        else:
            # Office / EPUB / anything else markitdown understands.
            from src.markitdown_runtime import is_markitdown_format, convert_to_markdown
            if is_markitdown_format(path):
                text = convert_to_markdown(path) or ""
            else:
                # Unknown binary-ish format: try as text, give up on failure.
                text = _read_text_with_fallback(path)
                if "\x00" in text[:1000]:
                    return "", "unsupported"
    except Exception as e:
        logger.warning("Project file extraction failed for %s: %s", filename, e)
        return "", "failed"

    text = (text or "").strip()
    if not text:
        return "", "empty"
    if len(text) > PROJECT_FILE_EXTRACT_CAP:
        text = text[:PROJECT_FILE_EXTRACT_CAP] + "\n[truncated]"
    return text, "ok"


# ----------------------------------------------------------------------
# Vector indexing
# ----------------------------------------------------------------------

def index_project_file(pf: Any, rag_manager: Any) -> bool:
    """Chunk + index one ProjectFile's extracted text into the vector store.

    Indexes at upload time even while the project is under the inline budget,
    so crossing the budget later flips to RAG mode with no lazy re-index
    inside a chat request. Returns True on success."""
    if rag_manager is None or not getattr(rag_manager, "healthy", False):
        return False
    if pf.extract_status != "ok" or not pf.extracted_text:
        return False

    owner_key = project_owner_key(pf.owner, pf.project_id)
    source = project_file_source(pf.project_id, pf.id)
    chunks = rag_manager._split_into_chunks(
        pf.extracted_text, PROJECT_CHUNK_SIZE, PROJECT_CHUNK_OVERLAP
    )
    if not chunks:
        return False

    docs = []
    for i, chunk in enumerate(chunks):
        docs.append((chunk, {
            "owner": owner_key,
            "doc_type": "project_file",
            "project_id": pf.project_id,
            "file_id": pf.id,
            "filename": pf.filename,
            "source": source,
            "chunk_index": i,
            "doc_id": project_chunk_doc_id(owner_key, source, i),
        }))
    try:
        result = rag_manager.add_documents_batch(docs)
        return bool(result and result.get("success"))
    except Exception as e:
        logger.warning("Project file indexing failed for %s: %s", pf.filename, e)
        return False


def unindex_project_file(project_id: str, file_id: str, rag_manager: Any) -> int:
    """Best-effort removal of one file's chunks. Orphans are unreachable
    anyway (retrieval filters on a project that no longer lists the file),
    so a Chroma outage here is logged, not fatal."""
    if rag_manager is None or not getattr(rag_manager, "healthy", False):
        return 0
    try:
        return rag_manager.delete_by_source(project_file_source(project_id, file_id))
    except Exception as e:
        logger.warning("Project chunk cleanup failed for %s/%s: %s", project_id, file_id, e)
        return 0


# ----------------------------------------------------------------------
# Save / delete
# ----------------------------------------------------------------------

def save_project_file(db, project: Any, owner: Optional[str], upload: Any,
                      upload_handler: Any, rag_manager: Any) -> Tuple[Any, bool]:
    """Validate, store, extract, and index an uploaded project file.

    Returns (ProjectFile row, is_duplicate). Caller commits. Raises
    ValueError with a user-facing message on validation failure."""
    import uuid
    from core.database import ProjectFile

    raw_name = upload.filename or "file"
    filename = secure_filename(raw_name)
    file_obj = upload.file

    # Size check (same 10MB policy as chat uploads).
    file_obj.seek(0, os.SEEK_END)
    size = file_obj.tell()
    file_obj.seek(0)
    max_size = getattr(upload_handler, "max_upload_size", 10 * 1024 * 1024)
    if size > max_size:
        raise ValueError(f"File exceeds {max_size // (1024 * 1024)}MB limit")
    if size == 0:
        raise ValueError("Empty file")

    mime = upload_handler.detect_content_type(file_obj, filename)
    if not upload_handler.is_safe_file_type(mime, filename):
        raise ValueError("File type not allowed")

    file_hash = upload_handler.calculate_file_hash(file_obj)

    # Dedup within this project only — cross-project sharing would couple
    # file lifecycles across projects.
    existing = (
        db.query(ProjectFile)
        .filter(ProjectFile.project_id == project.id,
                ProjectFile.file_hash == file_hash)
        .first()
    )
    if existing:
        return existing, True

    file_id = uuid.uuid4().hex
    _, ext = os.path.splitext(filename.lower())
    dest = os.path.join(project_files_dir(project.id), f"{file_id}{ext}")
    file_obj.seek(0)
    with open(dest, "wb") as out:
        shutil.copyfileobj(file_obj, out)

    text, status = extract_project_file_text(dest, filename)

    pf = ProjectFile(
        id=file_id,
        project_id=project.id,
        owner=owner,
        filename=filename,
        stored_path=dest,
        mime=mime,
        size_bytes=size,
        file_hash=file_hash,
        extracted_text=text or None,
        extracted_chars=len(text),
        extract_status=status,
        indexed=False,
    )
    pf.indexed = index_project_file(pf, rag_manager)
    db.add(pf)
    return pf, False


def delete_project_file_assets(pf: Any, rag_manager: Any) -> None:
    """Remove one file's vector chunks and binary. The DB row (and with it
    the extracted text) is the caller's to delete — keeps the transaction
    boundary in the route."""
    unindex_project_file(pf.project_id, pf.id, rag_manager)
    try:
        if pf.stored_path and inside_project_files(pf.stored_path) and os.path.exists(pf.stored_path):
            os.remove(pf.stored_path)
    except Exception as e:
        logger.warning("Failed to remove project file binary %s: %s", pf.stored_path, e)


def delete_project_assets(project: Any, rag_manager: Any) -> None:
    """Remove all of a project's vector chunks and its binaries directory."""
    for pf in list(project.files or []):
        unindex_project_file(pf.project_id, pf.id, rag_manager)
    dir_path = project_files_dir(project.id, create=False)
    if inside_project_files(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)
