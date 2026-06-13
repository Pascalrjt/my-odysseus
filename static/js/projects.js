/**
 * Projects — group chats around shared files and a shared system prompt.
 *
 * Renders as a dedicated fullscreen page (#projects-page in index.html) with
 * two views: a grid of project cards at /projects, and a per-project detail
 * view at /projects/<id> (chats column + files/instructions sidebar). This
 * module owns all behavior: project CRUD, file upload/delete (drag-drop +
 * browse), the inline/RAG context status line, the chats-in-project list,
 * "new chat in project" (which goes through the pending-chat flow in
 * sessions.js — the server session is only created on the first message),
 * and URL/history handling (pushState on navigation, popstate to restore).
 *
 * Exposed as window.projectsModule so sessions.js (move-to-project submenu,
 * chat header badge) can reach it without an import cycle.
 */
import * as uiModule from './ui.js';
import { styledConfirm } from './ui.js';
import { createSessionItem } from './sessions.js';

const API_BASE = window.location.origin;
// Mirrors PROJECT_INLINE_BUDGET in src/project_context.py — display only.
const INLINE_BUDGET = 30000;

let _projectsCache = null;     // [{id, name, ...}] or null = not loaded
let _currentProjectId = null;  // non-null while detail view is open
let _saveTimer = null;
let _pageOpen = false;
let _returnHash = '';          // chat hash to restore when the page closes
let _searchQuery = '';

const el = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

export async function getProjects(force = false) {
  if (_projectsCache && !force) return _projectsCache;
  try {
    const res = await fetch(`${API_BASE}/api/projects`);
    if (!res.ok) return _projectsCache || [];
    _projectsCache = await res.json();
  } catch (_) {
    return _projectsCache || [];
  }
  return _projectsCache;
}

function _invalidate() { _projectsCache = null; }

async function _getProject(pid) {
  const res = await fetch(`${API_BASE}/api/projects/${encodeURIComponent(pid)}`);
  if (!res.ok) throw new Error(`Project load failed (${res.status})`);
  return res.json();
}

export async function refreshCurrentProject() {
  _invalidate();
  if (_pageOpen && _currentProjectId) {
    await _showDetail(_currentProjectId);
  }
}

// ---------------------------------------------------------------------------
// Open / close / URL
// ---------------------------------------------------------------------------

function _setUrl(projectId) {
  const target = projectId ? `/projects/${encodeURIComponent(projectId)}` : '/projects';
  // Equality guard: when the route opener fires on a direct /projects load
  // the URL is already right — pushing again would double the history entry.
  if (window.location.pathname !== target) {
    history.pushState({}, '', target);
  }
}

export async function openProjects(projectId = null) {
  const page = el('projects-page');
  if (!page) return;
  if (!_pageOpen) {
    _pageOpen = true;
    // Where close returns to: the current chat hash — unless we landed
    // directly on a /projects URL, where there is no prior chat state.
    _returnHash = window.location.pathname.startsWith('/projects') ? '' : window.location.hash;
    page.classList.remove('hidden');
    window._collapseSidebarToRail?.();
  }
  _setUrl(projectId);
  if (projectId) {
    await _showDetail(projectId);
  } else {
    await _showList();
  }
}

export function closeProjectsPage({ navigate = true } = {}) {
  if (!_pageOpen) return;
  _pageOpen = false;
  _currentProjectId = null;
  document.querySelectorAll('.project-session-dropdown, .project-session-submenu').forEach(d => d.remove());
  el('projects-page')?.classList.add('hidden');
  window._restoreSidebarIfRouteCollapsed?.();
  if (navigate && window.location.pathname.startsWith('/projects')) {
    history.pushState({}, '', '/' + _returnHash);
  }
}

export function isProjectsOpen() { return _pageOpen; }

// ---------------------------------------------------------------------------
// List view
// ---------------------------------------------------------------------------

function _switchView(detail) {
  el('projects-list-view')?.classList.toggle('hidden', detail);
  el('project-detail-view')?.classList.toggle('hidden', !detail);
}

function _relTime(iso) {
  if (!iso) return '';
  // updated_at is naive-UTC isoformat (core/database.py) — pin the zone so
  // the browser doesn't parse it as local time.
  const t = new Date(/[zZ]|[+-]\d\d:\d\d$/.test(iso) ? iso : iso + 'Z').getTime();
  if (!Number.isFinite(t)) return '';
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

async function _showList() {
  _currentProjectId = null;
  _switchView(false);
  const list = el('projects-list');
  if (!list) return;
  list.innerHTML = '<div class="projects-empty">Loading…</div>';
  const projects = await getProjects(true);
  _renderList(projects);
}

function _renderList(projects) {
  const list = el('projects-list');
  if (!list) return;
  list.innerHTML = '';
  const q = _searchQuery.trim().toLowerCase();
  const visible = q
    ? (projects || []).filter(p =>
        (p.name || '').toLowerCase().includes(q) ||
        (p.description || '').toLowerCase().includes(q))
    : (projects || []);
  if (!visible.length) {
    list.innerHTML = `<div class="projects-empty">${q
      ? 'No projects match your search.'
      : 'No projects yet. Create one to share files and a system prompt across chats.'}</div>`;
    return;
  }
  visible.forEach(p => {
    const card = document.createElement('div');
    card.className = 'project-card';
    const desc = (p.description || '').trim();
    const rel = _relTime(p.updated_at);
    card.innerHTML = `
      <div class="project-card-title"></div>
      <div class="project-card-desc"></div>
      <div class="project-card-meta">${p.chat_count} chat${p.chat_count === 1 ? '' : 's'} · ${p.file_count} file${p.file_count === 1 ? '' : 's'}${rel ? ' · ' + rel : ''}</div>`;
    card.querySelector('.project-card-title').textContent = p.name;
    card.querySelector('.project-card-desc').textContent = desc || 'No description';
    card.addEventListener('click', () => openProjects(p.id));
    list.appendChild(card);
  });
}

// ---------------------------------------------------------------------------
// Detail view
// ---------------------------------------------------------------------------

function _autosizeDesc() {
  const desc = el('project-description-input');
  if (!desc) return;
  desc.style.height = 'auto';
  desc.style.height = desc.scrollHeight + 'px';
}

async function _showDetail(pid) {
  let data;
  try {
    data = await _getProject(pid);
  } catch (e) {
    uiModule.showError(String(e.message || e));
    // The detail URL is dead (bad/foreign id) — repair it before falling back.
    if (window.location.pathname.startsWith('/projects/')) {
      history.replaceState({}, '', '/projects');
    }
    return _showList();
  }
  _currentProjectId = pid;
  _switchView(true);
  el('project-name-input').value = data.name || '';
  el('project-description-input').value = data.description || '';
  el('project-system-prompt-input').value = data.system_prompt || '';
  el('project-save-status').textContent = '';
  _autosizeDesc();
  _renderFiles(data.files || []);
  _renderChats(data.chats || []);
}

// ---------------------------------------------------------------------------
// Detail view: files
// ---------------------------------------------------------------------------

function _fmtSize(bytes) {
  if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  if (bytes >= 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return bytes + ' B';
}

function _renderFiles(files) {
  const wrap = el('project-files-list');
  if (!wrap) return;
  wrap.innerHTML = '';
  let totalChars = 0;
  files.forEach(f => {
    if (f.extract_status === 'ok') totalChars += (f.extracted_chars || 0);
    const row = document.createElement('div');
    row.className = 'project-file-row';
    const chars = f.extract_status === 'ok'
      ? `${(f.extracted_chars / 1000).toFixed(1)}K chars`
      : f.extract_status;
    row.innerHTML = `
      <span class="project-file-dot ${f.indexed ? 'indexed' : ''}" title="${f.indexed ? 'Searchable' : 'Not indexed'}"></span>
      <a class="project-file-name" href="${API_BASE}/api/projects/${encodeURIComponent(_currentProjectId)}/files/${encodeURIComponent(f.id)}" target="_blank" rel="noopener"></a>
      <span class="project-file-meta">${_fmtSize(f.size_bytes || 0)} · ${chars}</span>
      <button class="project-file-delete" title="Remove file">✖</button>`;
    row.querySelector('.project-file-name').textContent = f.filename;
    row.querySelector('.project-file-delete').addEventListener('click', async () => {
      const ok = await styledConfirm(`Remove "${f.filename}" from this project? Chats keep their history but lose access to this file.`, { confirmText: 'Remove', danger: true });
      if (!ok) return;
      await fetch(`${API_BASE}/api/projects/${encodeURIComponent(_currentProjectId)}/files/${encodeURIComponent(f.id)}`, { method: 'DELETE' });
      _invalidate();
      _showDetail(_currentProjectId);
    });
    wrap.appendChild(row);
  });

  const status = el('project-context-status');
  if (status) {
    if (!files.length) {
      status.textContent = '';
    } else if (totalChars <= INLINE_BUDGET) {
      status.textContent = `${(totalChars / 1000).toFixed(1)}K / ${INLINE_BUDGET / 1000}K chars — files inlined into every chat`;
    } else {
      status.textContent = `${(totalChars / 1000).toFixed(1)}K chars — over inline budget, chats search the files instead`;
    }
  }
}

async function _uploadFiles(fileList) {
  if (!_currentProjectId || !fileList || !fileList.length) return;
  for (const file of fileList) {
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res = await fetch(`${API_BASE}/api/projects/${encodeURIComponent(_currentProjectId)}/files`, {
        method: 'POST', body: fd,
      });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        uiModule.showError(`${file.name}: ${payload.detail || 'upload failed'}`);
      } else {
        const out = await res.json();
        if (out.is_duplicate) uiModule.showToast(`${file.name} is already in this project`);
      }
    } catch (e) {
      uiModule.showError(`${file.name}: ${e}`);
    }
  }
  _invalidate();
  await _showDetail(_currentProjectId);
}

// ---------------------------------------------------------------------------
// Detail view: chats
// ---------------------------------------------------------------------------

function _renderChats(chats) {
  const wrap = el('project-chats-list');
  if (!wrap) return;
  wrap.innerHTML = '';
  document.querySelectorAll('.project-session-dropdown, .project-session-submenu').forEach(d => d.remove());
  if (!chats.length) {
    wrap.innerHTML = '<div class="projects-empty">No chats yet — start one above, or move an existing chat here from its ⋮ menu.</div>';
    return;
  }
  chats.forEach(c => {
    const row = createSessionItem(c);
    row.classList.add('project-chat-row');
    row._sessionDropdown?.classList.add('project-session-dropdown');
    row._sessionDropdownSubmenus?.forEach(sub => sub.classList.add('project-session-submenu'));
    if (window.sessionModule?.getCurrentSessionId?.() === c.id) {
      row.classList.add('active-session');
    }
    wrap.appendChild(row);
  });
}

async function _newChatInProject() {
  if (!_currentProjectId) return;
  let dc = null;
  try {
    dc = await (await fetch(`${API_BASE}/api/default-chat`)).json();
  } catch (_) { /* fall through */ }
  if (!dc || !dc.endpoint_url || !dc.model) {
    uiModule.showError('No default model configured — pick a model from the chat screen first.');
    return;
  }
  // closeProjectsPage clears _currentProjectId — capture it before closing.
  const pid = _currentProjectId;
  closeProjectsPage();
  // Pending-chat flow: the session is created on first message, carrying
  // project_id via materializePendingSession().
  window.sessionModule?.createDirectChat?.(dc.endpoint_url, dc.model, dc.endpoint_id, pid);
}

// ---------------------------------------------------------------------------
// Detail view: settings save / delete
// ---------------------------------------------------------------------------

async function _saveProject() {
  if (!_currentProjectId) return;
  const fd = new FormData();
  fd.append('name', el('project-name-input').value.trim() || 'Untitled project');
  fd.append('description', el('project-description-input').value);
  fd.append('system_prompt', el('project-system-prompt-input').value);
  const res = await fetch(`${API_BASE}/api/projects/${encodeURIComponent(_currentProjectId)}`, {
    method: 'PATCH', body: fd,
  });
  const status = el('project-save-status');
  if (res.ok) {
    _invalidate();
    if (status) {
      status.textContent = 'Saved';
      clearTimeout(_saveTimer);
      _saveTimer = setTimeout(() => { status.textContent = ''; }, 2000);
    }
  } else if (status) {
    status.textContent = 'Save failed';
  }
}

async function _deleteProject() {
  if (!_currentProjectId) return;
  const ok = await styledConfirm(
    'Delete this project? Its files are removed permanently. Chats survive as normal chats with their full history.',
    { confirmText: 'Delete project', danger: true },
  );
  if (!ok) return;
  await fetch(`${API_BASE}/api/projects/${encodeURIComponent(_currentProjectId)}`, { method: 'DELETE' });
  _invalidate();
  // Sidebar list + badge may reference the deleted project.
  window.sessionModule?.loadSessions?.().catch?.(() => {});
  // The detail URL is dead — replace rather than push.
  if (window.location.pathname.startsWith('/projects/')) {
    history.replaceState({}, '', '/projects');
  }
  await _showList();
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

export function init() {
  const page = el('projects-page');
  if (!page) return;

  el('projects-page-close')?.addEventListener('click', () => closeProjectsPage());
  el('projects-detail-close')?.addEventListener('click', () => closeProjectsPage());
  el('project-back-btn')?.addEventListener('click', () => openProjects(null));

  el('projects-search')?.addEventListener('input', (e) => {
    _searchQuery = e.target.value || '';
    if (_projectsCache) _renderList(_projectsCache);
  });

  el('new-project-btn')?.addEventListener('click', async () => {
    const name = await uiModule.styledPrompt('Name this project:', {
      title: 'New project',
      placeholder: 'e.g. Thesis, Apartment hunt, Q3 launch',
      confirmText: 'Create',
    });
    if (!name || !name.trim()) return;
    const fd = new FormData();
    fd.append('name', name.trim());
    const res = await fetch(`${API_BASE}/api/projects`, { method: 'POST', body: fd });
    if (!res.ok) {
      uiModule.showError('Failed to create project');
      return;
    }
    _invalidate();
    const project = await res.json();
    await openProjects(project.id);
  });

  el('project-save-btn')?.addEventListener('click', _saveProject);
  el('project-delete-btn')?.addEventListener('click', _deleteProject);
  el('project-new-chat-btn')?.addEventListener('click', _newChatInProject);

  // Title/description save on commit (blur/Enter); the Save button covers
  // the instructions textarea.
  el('project-name-input')?.addEventListener('change', _saveProject);
  el('project-description-input')?.addEventListener('change', _saveProject);
  el('project-description-input')?.addEventListener('input', _autosizeDesc);

  // File upload: browse + drag-drop
  el('project-file-browse')?.addEventListener('click', () => el('project-file-input')?.click());
  el('project-file-input')?.addEventListener('change', (e) => {
    _uploadFiles(Array.from(e.target.files || []));
    e.target.value = '';
  });
  const dz = el('project-dropzone');
  if (dz) {
    dz.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
      dz.classList.add('dragging');
    });
    dz.addEventListener('dragleave', () => dz.classList.remove('dragging'));
    dz.addEventListener('drop', (e) => {
      e.preventDefault();
      dz.classList.remove('dragging');
      _uploadFiles(Array.from(e.dataTransfer.files || []));
    });
  }

  // Browser back/forward across /projects, /projects/<id>, and the chat
  // view. Hash-only popstates (session switches) fall through to the
  // closed-page no-op so the hashchange listener in sessions.js owns them.
  window.addEventListener('popstate', () => {
    const path = window.location.pathname;
    if (path === '/projects') {
      if (_pageOpen) _showList(); else openProjects();
      return;
    }
    const m = path.match(/^\/projects\/([^/]+)$/);
    if (m) {
      const pid = decodeURIComponent(m[1]);
      if (_pageOpen) _showDetail(pid); else openProjects(pid);
      return;
    }
    if (_pageOpen) closeProjectsPage({ navigate: false });
  });

  // Expose for sessions.js (badge, move-to-project submenu) and app.js
  // (sidebar/rail button, route opener) — no import cycle.
  window.projectsModule = { getProjects, openProjects, closeProjectsPage, refreshCurrentProject, isProjectsOpen, init };
}

export default { init, openProjects, closeProjectsPage, refreshCurrentProject, isProjectsOpen, getProjects };
