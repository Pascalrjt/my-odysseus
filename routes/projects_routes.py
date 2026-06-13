# routes/projects_routes.py
"""Projects API — projects group chat sessions and give them a shared
system prompt plus shared project files (see src/project_files.py).

Deletion semantics (the load-bearing part):
- Deleting a FILE removes its vector chunks, binary, and row. Existing
  transcripts are untouched — chat prefaces are assembled per-turn and never
  persisted — so future turns simply no longer see the file.
- Deleting a PROJECT detaches its sessions (DB UPDATE + in-memory cache,
  belt-and-braces over the FK's ON DELETE SET NULL) and removes all file
  assets. Chats survive as normal chats with full history.
"""
import logging
import os
import uuid

from fastapi import APIRouter, Form, HTTPException, Request, File, UploadFile
from fastapi.responses import FileResponse

from core.database import Project, ProjectFile, Session as DbSession, SessionLocal
from src.auth_helpers import effective_user, owner_filter
from src.project_files import (
    delete_project_assets,
    delete_project_file_assets,
    index_project_file,
    inside_project_files,
    save_project_file,
)

logger = logging.getLogger(__name__)


def setup_projects_routes(session_manager, upload_handler, rag_manager=None):
    """Setup project routes. rag_manager may be None at startup (ChromaDB
    down/lazy) — we re-resolve through the singleton per request.

    The router is created here (not at module level) so repeated setup calls
    — e.g. one per test — never stack duplicate routes bound to stale
    dependencies."""
    router = APIRouter(prefix="/api/projects", tags=["projects"])

    def _rag():
        if rag_manager is not None:
            return rag_manager
        try:
            from src.rag_singleton import get_rag_manager
            return get_rag_manager()
        except Exception:
            return None

    def _get_project_or_404(db, request: Request, pid: str) -> Project:
        """404 on missing OR foreign — don't leak existence (same convention
        as _verify_session_owner in session_routes)."""
        user = effective_user(request)
        project = db.query(Project).filter(Project.id == pid).first()
        if project is None:
            raise HTTPException(404, "Project not found")
        if user and project.owner and project.owner != user:
            raise HTTPException(404, "Project not found")
        return project

    def _project_summary(db, project: Project) -> dict:
        d = project.to_dict()
        d["file_count"] = (
            db.query(ProjectFile).filter(ProjectFile.project_id == project.id).count()
        )
        d["chat_count"] = (
            db.query(DbSession)
            .filter(DbSession.project_id == project.id, DbSession.archived == False)  # noqa: E712
            .count()
        )
        return d

    @router.get("")
    def list_projects(request: Request):
        user = effective_user(request)
        db = SessionLocal()
        try:
            q = db.query(Project).order_by(Project.updated_at.desc())
            q = owner_filter(q, Project, user)
            return [_project_summary(db, p) for p in q.all()]
        finally:
            db.close()

    @router.post("")
    def create_project(
        request: Request,
        name: str = Form(...),
        description: str = Form(""),
        system_prompt: str = Form(""),
    ):
        name = (name or "").strip()
        if not name:
            raise HTTPException(400, "Project name is required")
        user = effective_user(request)
        db = SessionLocal()
        try:
            project = Project(
                id=uuid.uuid4().hex,
                name=name[:200],
                description=description or "",
                system_prompt=system_prompt or "",
                owner=user,
            )
            db.add(project)
            db.commit()
            return _project_summary(db, project)
        finally:
            db.close()

    @router.get("/{pid}")
    def get_project(request: Request, pid: str):
        db = SessionLocal()
        try:
            project = _get_project_or_404(db, request, pid)
            files = (
                db.query(ProjectFile)
                .filter(ProjectFile.project_id == pid)
                .order_by(ProjectFile.created_at.asc())
                .all()
            )
            user = effective_user(request)
            chats_q = db.query(DbSession).filter(
                DbSession.project_id == pid,
                DbSession.archived == False,  # noqa: E712
            )
            chats_q = owner_filter(chats_q, DbSession, user)
            chats = chats_q.order_by(DbSession.last_accessed.desc()).all()
            d = project.to_dict()
            d["files"] = [f.to_dict() for f in files]
            d["chats"] = [{
                "id": s.id,
                "name": s.name,
                "model": s.model,
                "endpoint_url": s.endpoint_url,
                "rag": s.rag,
                "archived": s.archived,
                "folder": s.folder,
                "is_important": s.is_important or False,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None,
                "project_id": s.project_id,
                "mode": s.mode,
                "message_count": s.message_count or 0,
                "last_message_at": s.last_message_at.isoformat() if s.last_message_at else None,
            } for s in chats]
            return d
        finally:
            db.close()

    @router.patch("/{pid}")
    def update_project(
        request: Request, pid: str,
        name: str = Form(None),
        description: str = Form(None),
        system_prompt: str = Form(None),
    ):
        db = SessionLocal()
        try:
            project = _get_project_or_404(db, request, pid)
            if name is not None:
                name = name.strip()
                if not name:
                    raise HTTPException(400, "Project name cannot be empty")
                project.name = name[:200]
            if description is not None:
                project.description = description
            if system_prompt is not None:
                project.system_prompt = system_prompt
            db.commit()
            return _project_summary(db, project)
        finally:
            db.close()

    @router.delete("/{pid}")
    def delete_project(request: Request, pid: str):
        db = SessionLocal()
        try:
            project = _get_project_or_404(db, request, pid)
            # 1. Vector chunks + binaries (best-effort, before the rows go).
            delete_project_assets(project, _rag())
            # 2. Detach sessions explicitly. The FK is ON DELETE SET NULL, but
            #    do it ourselves so correctness never depends on pragma state.
            db.query(DbSession).filter(DbSession.project_id == pid).update(
                {DbSession.project_id: None}, synchronize_session=False
            )
            # 3. Project row (ProjectFile rows cascade via the relationship).
            db.delete(project)
            db.commit()
        finally:
            db.close()
        # 4. Null project_id on EVERY cached session object — including
        #    metadata-only entries — because get_sessions_for_user() and the
        #    chat path serve from this cache; a stale cached project_id would
        #    survive the DB update. A turn already streaming keeps the context
        #    it loaded; the next turn sees None (and build_project_context
        #    returns empty for a missing project row as the final backstop).
        try:
            for sess in session_manager.sessions.values():
                if getattr(sess, "project_id", None) == pid:
                    sess.project_id = None
        except Exception:
            logger.warning("Failed to clear cached project_id for %s", pid, exc_info=True)
        return {"deleted": pid}

    @router.post("/{pid}/files")
    def upload_project_file(request: Request, pid: str, file: UploadFile = File(...)):
        user = effective_user(request)
        db = SessionLocal()
        try:
            project = _get_project_or_404(db, request, pid)
            try:
                pf, duplicate = save_project_file(
                    db, project, user, file, upload_handler, _rag()
                )
            except ValueError as e:
                raise HTTPException(400, str(e))
            db.commit()
            d = pf.to_dict()
            d["is_duplicate"] = duplicate
            return d
        finally:
            db.close()

    @router.delete("/{pid}/files/{fid}")
    def delete_project_file(request: Request, pid: str, fid: str):
        db = SessionLocal()
        try:
            _get_project_or_404(db, request, pid)
            pf = (
                db.query(ProjectFile)
                .filter(ProjectFile.id == fid, ProjectFile.project_id == pid)
                .first()
            )
            if pf is None:
                raise HTTPException(404, "File not found")
            delete_project_file_assets(pf, _rag())
            db.delete(pf)  # extracted text dies with the row
            db.commit()
            return {"deleted": fid}
        finally:
            db.close()

    @router.get("/{pid}/files/{fid}")
    def download_project_file(request: Request, pid: str, fid: str):
        db = SessionLocal()
        try:
            _get_project_or_404(db, request, pid)
            pf = (
                db.query(ProjectFile)
                .filter(ProjectFile.id == fid, ProjectFile.project_id == pid)
                .first()
            )
            if pf is None:
                raise HTTPException(404, "File not found")
            path = pf.stored_path
            if not path or not inside_project_files(path) or not os.path.isfile(path):
                raise HTTPException(404, "File not found")
            return FileResponse(
                path,
                media_type=pf.mime or "application/octet-stream",
                filename=pf.filename,
                headers={"X-Content-Type-Options": "nosniff"},
            )
        finally:
            db.close()

    @router.post("/{pid}/reindex")
    def reindex_project(request: Request, pid: str):
        """Re-index files whose chunks never made it into the vector store
        (ChromaDB was down at upload time)."""
        db = SessionLocal()
        try:
            _get_project_or_404(db, request, pid)
            rag = _rag()
            if rag is None or not getattr(rag, "healthy", False):
                raise HTTPException(503, "Vector store unavailable")
            pending = (
                db.query(ProjectFile)
                .filter(ProjectFile.project_id == pid,
                        ProjectFile.indexed == False,  # noqa: E712
                        ProjectFile.extract_status == "ok")
                .all()
            )
            done = 0
            for pf in pending:
                if index_project_file(pf, rag):
                    pf.indexed = True
                    done += 1
            db.commit()
            return {"reindexed": done, "pending": len(pending) - done}
        finally:
            db.close()

    return router
