"""Regression guards for project chat navigation state.

The affected code is browser-only ES modules, so these tests inspect the
source for the small routing/refresh invariants that keep project pages and
chat selection from diverging.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SESSIONS_JS = ROOT / "static/js/sessions.js"
PROJECTS_JS = ROOT / "static/js/projects.js"


def _function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    rest = source[start + len(signature):]
    match = re.search(r"\nexport (?:async )?function |\nfunction ", rest)
    return rest[: match.start()] if match else rest


def test_select_session_closes_projects_and_uses_root_chat_route():
    body = _function_body(
        SESSIONS_JS.read_text(encoding="utf-8"),
        "export async function selectSession(id, { keepSidebar = false } = {})",
    )
    assert "window.projectsModule?.isProjectsOpen?.()" in body
    assert "window.projectsModule.closeProjectsPage({ navigate: false })" in body
    assert "const targetUrl = '/#' + id" in body
    assert "history.replaceState(null, '', targetUrl)" in body
    assert "history.replaceState(null, '', '/')" in body


def test_materialize_pending_project_chat_refreshes_project_state():
    body = _function_body(
        SESSIONS_JS.read_text(encoding="utf-8"),
        "export async function materializePendingSession()",
    )
    assert "fd.append('project_id', pending.projectId)" in body
    assert "history.replaceState(null, '', '/#' + payload.id)" in body
    assert "payload.project_id" in body
    assert "window.projectsModule?.refreshCurrentProject?.()" in body


def test_projects_module_exposes_refresh_current_project():
    text = PROJECTS_JS.read_text(encoding="utf-8")
    assert "export async function refreshCurrentProject()" in text
    assert "_invalidate();" in _function_body(text, "export async function refreshCurrentProject()")
    assert "window.projectsModule = { getProjects, openProjects, closeProjectsPage, refreshCurrentProject" in text
    assert "export default { init, openProjects, closeProjectsPage, refreshCurrentProject" in text


def test_project_chats_reuse_sidebar_session_item_renderer():
    projects = PROJECTS_JS.read_text(encoding="utf-8")
    sessions = SESSIONS_JS.read_text(encoding="utf-8")
    body = _function_body(projects, "function _renderChats(chats)")
    assert "import { createSessionItem } from './sessions.js';" in projects
    assert "const row = createSessionItem(c)" in body
    assert "project-chat-name" not in body
    assert "export function createSessionItem(s)" in sessions
    assert "document.querySelectorAll(`.list-item[data-session-id=\"${id}\"]`)" in sessions
