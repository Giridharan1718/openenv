"""agent_tools.py — structured tool interface for agentic debugging."""
from __future__ import annotations

import os
from typing import Any

from .language_executors import get_executor
from .safety import is_safe


def view_file(workspace: str, path: str) -> dict[str, Any]:
    full = os.path.abspath(os.path.join(workspace, path))
    if not full.startswith(os.path.abspath(workspace)):
        return {"error": "path escapes workspace"}
    try:
        with open(full, "r", encoding="utf-8") as f:
            return {"path": path, "content": f.read()}
    except Exception as e:
        return {"error": str(e)}


def run_tests(code: str, language: str, entry_point: str | None, tests: list[dict]) -> dict[str, Any]:
    exe = get_executor(language)
    if not (hasattr(exe, "available") and exe.available()):
        return {"error": f"{language} runtime unavailable", "results": []}
    ok, err = exe.syntax_check(code)
    if not ok:
        return {"syntax_ok": False, "error": err, "results": []}
    res = exe.run_tests(code, entry_point, tests)
    passed = sum(1 for r in res if r.get("passed"))
    return {"syntax_ok": True, "passed": passed, "total": len(res), "results": res}


def apply_patch(original: str, patch_text: str) -> dict[str, Any]:
    # minimal: if patch_text is full code, replace; validate safety
    candidate = patch_text if "def " in patch_text or "function" in patch_text else None
    if candidate is None:
        return {"error": "patch must contain replacement code"}
    safe, reasons = is_safe(candidate)
    if not safe:
        return {"error": "unsafe patch", "reasons": reasons}
    return {"applied": True, "code": candidate}


def get_task_info(task) -> dict[str, Any]:
    return {
        "task_id": getattr(task, "task_id", None),
        "description": getattr(task, "description", ""),
        "error_type": getattr(task, "error_type", ""),
        "difficulty": getattr(task, "difficulty", ""),
        "language": getattr(task, "language", "python"),
        "hints": list(getattr(task, "hints", [])),
        "max_steps": getattr(task, "max_steps", 3),
    }
PYEOF