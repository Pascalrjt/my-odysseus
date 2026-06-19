"""Regression guards for the "Document Mode" chat appearance toggle.

Document mode swaps AI chat bubbles for a full-width document layout. It lives
in Settings -> Appearance -> Chat Area and rides the existing `data-ui-key`
visibility system (localStorage-backed, applied via app.js `applyUIVis`). The
code is browser-only (HTML + ES module + CSS), so these tests inspect the source
for the wiring invariants: the toggle exists in the Appearance panel's Chat Area
section, defaults OFF, maps to the `doc-chat` body class, and the CSS only
flattens AI chrome (never the user bubble).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "static/index.html"
APP_JS = ROOT / "static/app.js"
STYLE_CSS = ROOT / "static/style.css"


def _appearance_panel(html: str) -> str:
    start = html.index('data-settings-panel="appearance"')
    rest = html[start:]
    nxt = rest.find('data-settings-panel=', 1)
    return rest[:nxt] if nxt != -1 else rest


def test_document_mode_toggle_in_appearance_chat_area():
    html = INDEX_HTML.read_text(encoding="utf-8")
    panel = _appearance_panel(html)
    assert 'data-ui-key="doc-chat"' in panel
    assert "Document Mode" in panel
    # Default OFF: the checkbox must not be pre-checked.
    assert 'data-ui-key="doc-chat"' in panel
    assert 'checked data-ui-key="doc-chat"' not in panel
    # It must NOT linger in the old Brain (memory) settings tab.
    assert 'id="doc-chat-toggle"' not in html


def test_doc_chat_wired_in_apply_ui_vis():
    js = APP_JS.read_text(encoding="utf-8")
    # Default-OFF on first run.
    assert "'doc-chat'" in js
    assert "UI_VIS_DEFAULT_OFF" in js
    # Applied as a body class, on only when explicitly true.
    assert "classList.toggle('doc-chat', state['doc-chat'] === true)" in js


def test_doc_chat_css_flattens_ai_chrome_only():
    css = STYLE_CSS.read_text(encoding="utf-8")
    assert "body.doc-chat .msg-ai" in css
    assert "body.doc-chat .thinking-section" in css
    assert "body.doc-chat .agent-thread" in css
    assert "body.doc-chat .agent-tool-output" in css
    assert "body.doc-chat .sources-section" in css
    # The user bubble must never be flattened by document mode.
    assert "body.doc-chat .msg-user" not in css
