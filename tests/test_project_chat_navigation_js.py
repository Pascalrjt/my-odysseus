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
STYLE_CSS = ROOT / "static/style.css"


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


def test_load_sessions_refreshes_active_project_metadata():
    body = _function_body(
        SESSIONS_JS.read_text(encoding="utf-8"),
        "export async function loadSessions()",
    )
    same_session = body[body.index("} else if (targetId && targetId === currentSessionId) {"):]
    same_session = same_session[:same_session.index("\n    }\n\n    // No session selected")]
    assert "_updateProjectBadge(s)" in same_session
    assert "if (s && s.project_id)" in same_session
    assert "window.projectsModule?.refreshCurrentProject?.()" in same_session


def test_pending_project_chat_survives_model_picker_switch():
    sessions = SESSIONS_JS.read_text(encoding="utf-8")
    body = _function_body(sessions, "export function setPendingChat(next)")
    assert "Object.prototype.hasOwnProperty.call(next, 'projectId')" in body
    assert "_pendingChat && _pendingChat.projectId" in body
    assert "_pendingChat = { ...next, projectId: projectId || null }" in body
    assert "setPendingChat," in sessions
    assert "setPendingChat: (v) => { _pendingChat = v; }" not in sessions


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


def test_projects_page_hides_global_model_picker():
    projects = PROJECTS_JS.read_text(encoding="utf-8")
    styles = STYLE_CSS.read_text(encoding="utf-8")
    assert "function _setProjectsPageOpen(open)" in projects
    assert "document.body?.classList.toggle('projects-page-open', !!open)" in projects
    assert "el('model-picker-menu')?.classList.add('hidden')" in projects
    assert "_setProjectsPageOpen(true)" in _function_body(
        projects,
        "export async function openProjects(projectId = null)",
    )
    assert "_setProjectsPageOpen(false)" in _function_body(
        projects,
        "export function closeProjectsPage({ navigate = true } = {})",
    )
    assert "body.projects-page-open #model-picker-wrap" in styles
    assert "display: none !important;" in _function_body(
        styles,
        "body.projects-page-open #model-picker-wrap",
    )
