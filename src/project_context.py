"""Project context for chats — shared system prompt + project files.

Every chat in a project receives:
1. The project's system prompt — user-authored, so a TRUSTED plain system
   message (stacks with preset prompts; llm_core merges system messages).
2. A listing of project file names — file-derived, so untrusted-wrapped.
3. File CONTENT, hybrid:
   - total extracted text <= PROJECT_INLINE_BUDGET → inline everything
     (small projects get the whole corpus every turn, deterministic);
   - over budget → retrieve only relevant chunks per message from the
     vector store (files are indexed at upload time, see project_files.py).

The two trust zones are returned separately: the caller inserts the system
prompt next to the preset prompt and the file blocks AFTER the global
untrusted-context policy message.

Owner scoping: callers pass the SESSION's owner (not get_current_user(),
which is "api" for bearer-token calls). Retrieval uses the composite owner
key, which confines results to exactly this user's project.

Incognito note: project context injects in incognito chats by design — it is
explicit, user-configured session state, the same class as preset prompts
(which also inject in incognito). Retrieval is read-only; nothing is written.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.prompt_security import untrusted_context_message
from src.project_files import project_owner_key

logger = logging.getLogger(__name__)

# Total extracted chars across all project files below which everything is
# inlined into every turn. Mirrors the spirit of MAX_INLINE_ATTACHMENT_CHARS
# (24K) for per-message attachments, slightly higher since project files are
# the user's deliberate, curated corpus.
PROJECT_INLINE_BUDGET = 30_000
PROJECT_RAG_K = 5
PROJECT_RAG_THRESHOLD = 0.35  # matches ChatProcessor.RAG_SIMILARITY_THRESHOLD
PROJECT_RAG_MAX_CHARS = 10_000  # matches the personal-RAG block cap


def _format_size(chars: int) -> str:
    if chars >= 1000:
        return f"{chars / 1000:.1f}K chars"
    return f"{chars} chars"


def build_project_context(
    project_id: str,
    owner: Optional[str],
    message: str,
    rag_manager: Any = None,
) -> Tuple[Optional[Dict[str, str]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build the project portion of a chat preface.

    Returns (system_msg | None, context_msgs, rag_sources). All empty when
    the project is missing (deleted mid-session — the chat degrades to a
    normal chat) or owned by someone else (defense in depth; session PATCH
    already enforces ownership).
    """
    from core.database import Project, ProjectFile, SessionLocal

    db = SessionLocal()
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if project is None:
            return None, [], []
        if owner and project.owner and project.owner != owner:
            logger.warning("Project %s owner mismatch for session owner %r", project_id, owner)
            return None, [], []
        files = (
            db.query(ProjectFile)
            .filter(ProjectFile.project_id == project_id)
            .order_by(ProjectFile.created_at.asc())
            .all()
        )
        project_name = project.name
        system_prompt = (project.system_prompt or "").strip()
        # Detach the data we need before the DB session closes.
        file_infos = [{
            "filename": f.filename,
            "extracted_chars": f.extracted_chars or 0,
            "extract_status": f.extract_status,
            "extracted_text": f.extracted_text or "",
        } for f in files]
    finally:
        db.close()

    system_msg = {"role": "system", "content": system_prompt} if system_prompt else None
    context_msgs: List[Dict[str, Any]] = []
    rag_sources: List[Dict[str, Any]] = []

    if not file_infos:
        return system_msg, context_msgs, rag_sources

    # Always-on listing: even in RAG mode the model should know what exists.
    listing_lines = []
    for fi in file_infos:
        if fi["extract_status"] == "ok":
            listing_lines.append(f"- {fi['filename']} ({_format_size(fi['extracted_chars'])} extracted)")
        else:
            listing_lines.append(f"- {fi['filename']} (no extractable text)")
    context_msgs.append(untrusted_context_message(
        "project files",
        f"Files attached to project \"{project_name}\":\n" + "\n".join(listing_lines),
    ))

    readable = [fi for fi in file_infos if fi["extract_status"] == "ok" and fi["extracted_text"]]
    if not readable:
        return system_msg, context_msgs, rag_sources

    total = sum(fi["extracted_chars"] for fi in readable)

    if total <= PROJECT_INLINE_BUDGET:
        body = "\n\n---\n\n".join(
            f"[{fi['filename']}]\n{fi['extracted_text']}" for fi in readable
        )
        context_msgs.append(untrusted_context_message("project file contents", body))
        return system_msg, context_msgs, rag_sources

    # Over budget → retrieval mode.
    results = []
    if rag_manager is not None and getattr(rag_manager, "healthy", False):
        try:
            results = rag_manager.search(
                message, k=PROJECT_RAG_K,
                owner=project_owner_key(owner, project_id),
            )
        except Exception as e:
            logger.warning("Project RAG search failed for %s: %s", project_id, e)
            results = []
    else:
        # Vector store down: degrade to truncated inline heads so the model
        # still sees something, with an explicit notice.
        budget_each = max(PROJECT_INLINE_BUDGET // len(readable), 500)
        body = "\n\n---\n\n".join(
            f"[{fi['filename']}]\n{fi['extracted_text'][:budget_each]}" for fi in readable
        )
        context_msgs.append(untrusted_context_message(
            "project file contents (truncated)",
            "[Project files exceed the inline budget and search is currently "
            "unavailable — content below is truncated.]\n\n" + body,
        ))
        return system_msg, context_msgs, rag_sources

    relevant = [r for r in results if r.get("similarity", 0) >= PROJECT_RAG_THRESHOLD]
    if relevant:
        rag_sources = [{
            "filename": r["metadata"].get("filename", "unknown"),
            "snippet": r["document"][:200],
            "similarity": round(r.get("similarity", 0), 3),
        } for r in relevant]
        content = "Relevant excerpts from project files:\n\n" + "\n\n---\n\n".join(
            f"[{s['filename']}]\n{r['document']}" for s, r in zip(rag_sources, relevant)
        )
        if len(content) > PROJECT_RAG_MAX_CHARS:
            content = content[:PROJECT_RAG_MAX_CHARS] + "\n[Truncated]"
        context_msgs.append(untrusted_context_message("project file excerpts", content))

    return system_msg, context_msgs, rag_sources
