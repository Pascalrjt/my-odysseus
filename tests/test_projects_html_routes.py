"""The /projects and /projects/{project_id} HTML routes must serve the SPA.

app.py is too heavy to import in tests (it wires the full app at module
level), so — like test_app_static_mime.py — this inspects the source via ast.
"""

import ast
from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _get_route_handlers():
    """Map of @app.get path -> the decorated async function node."""
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))
    handlers = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if (
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Attribute)
                and dec.func.attr == "get"
                and isinstance(dec.func.value, ast.Name)
                and dec.func.value.id == "app"
                and dec.args
                and isinstance(dec.args[0], ast.Constant)
            ):
                handlers[dec.args[0].value] = node
    return handlers


def _returns_serve_index(fn_node):
    """True when the handler body is `return await serve_index(request)`."""
    for node in ast.walk(fn_node):
        if (
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Await)
            and isinstance(node.value.value, ast.Call)
            and isinstance(node.value.value.func, ast.Name)
            and node.value.value.func.id == "serve_index"
        ):
            return True
    return False


def test_projects_list_route_serves_spa():
    handlers = _get_route_handlers()
    assert "/projects" in handlers
    assert _returns_serve_index(handlers["/projects"])


def test_projects_detail_route_serves_spa():
    handlers = _get_route_handlers()
    assert "/projects/{project_id}" in handlers
    assert _returns_serve_index(handlers["/projects/{project_id}"])
