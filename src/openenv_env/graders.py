"""
graders.py — Scoring functions for the CodeDebug environment.

Every grader takes (action: Action, task: TaskSpec) and returns a float in [0.0, 1.0].
"""

from __future__ import annotations

import ast
import textwrap
import traceback
from types import ModuleType
from typing import Any

from .models import Action
from .tasks import TaskSpec

# ---------------------------------------------------------------------------
# Helper: execute code in a sandboxed namespace
# ---------------------------------------------------------------------------

def _run_code(source: str) -> tuple[ModuleType | None, str | None]:
    """
    Compile and exec *source* in a fresh namespace.

    Returns (namespace_dict, error_string).  error_string is None on success.
    """
    namespace: dict[str, Any] = {}
    try:
        compiled = compile(source, "<agent_code>", "exec")
        exec(compiled, namespace)  # noqa: S102
        return namespace, None
    except Exception:
        return None, traceback.format_exc(limit=3)




def _node_exe():
    import glob as _g
    import os as _o
    import shutil as _sh
    candidates = []
    w = _sh.which("node")
    if w:
        candidates.append(w)
    candidates += _g.glob(r"C:\Program Files\WindowsApps\*\app\resources\agent-payload\tools\node-*\node.exe")
    candidates += _g.glob(r"C:\Program Files\nodejs\node.exe")
    for c in candidates:
        if c and _o.path.exists(c):
            return c
    return "node"


_NODE = _node_exe()


def _run_node(args):
    import subprocess as _sp
    try:
        return _sp.run(args, capture_output=True, text=True, timeout=15)
    except PermissionError:
        import shlex as _shlex
        # args[0] is node binary; reconstruct without it
        rest = " ".join(_shlex.quote(a) for a in args[1:])
        return _sp.run(["C:/Program Files/Git/bin/bash.exe", "-c", "node " + rest],
                       capture_output=True, text=True, timeout=15)

def _js_output_grader(action: Action, task: TaskSpec) -> float:
    import json as _json
    import os as _os
    import tempfile as _tp

    code = textwrap.dedent(action.fixed_code)
    # Write code to a temp .mjs module
    with _tp.NamedTemporaryFile("w", suffix=".mjs", delete=False) as f:
        f.write(code)
        modpath = f.name
    try:
        # syntax check
        r = _run_node([_NODE, "--check", modpath])
        if r.returncode != 0:
            return 0.0
        ep = task.entry_point
        passed = 0
        total = len(task.test_inputs) or 1
        for test in task.test_inputs:
            args = test.get("args", [])
            expected = test.get("expected")
            script = (
                f"import {{ {ep} }} from 'file:///{modpath.replace(chr(92), '/') }';\n"
                f"const args = JSON.parse(process.argv[1]);\n"
                f"let r; try {{ r = {ep}(...args); }} catch(e) {{ r = null; }}\n"
                f"console.log(JSON.stringify(r));\n"
            )
            sfile = None
            with _tp.NamedTemporaryFile("w", suffix=".mjs", delete=False) as sf:
                sf.write(script)
                sfile = sf.name
            try:
                run = _run_node(
                    [_NODE, sfile, _json.dumps(list(args))],
                )
            finally:
                try:
                    _os.unlink(sfile)
                except OSError:
                    pass
            if run.returncode != 0:
                continue
            try:
                got = _json.loads(run.stdout.strip())
            except Exception:
                continue
            if got == expected:
                passed += 1
        return round(passed / total, 4)
    finally:
        try:
            _os.unlink(modpath)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Grader 1 — Syntax Correctness
# ---------------------------------------------------------------------------

def syntax_grader(action: Action, task: TaskSpec) -> float:
    """
    Award 1.0 if the agent's code parses cleanly, 0.0 otherwise.

    This is a fast, dependency-free check using the built-in ast module.
    """
    try:
        if getattr(task, "language", "python") == "javascript":
            # use node --check
            import os as _os
            import tempfile as _tp
            with _tp.NamedTemporaryFile("w", suffix=".mjs", delete=False) as f:
                f.write(textwrap.dedent(action.fixed_code))
                p = f.name
            try:
                r = _run_node([_NODE, "--check", p])
                return 1.0 if r.returncode == 0 else 0.0
            finally:
                try:
                    _os.unlink(p)
                except OSError:
                    pass
        ast.parse(textwrap.dedent(action.fixed_code))
        return 1.0
    except SyntaxError:
        return 0.0


# ---------------------------------------------------------------------------
# Grader 2 — Output Correctness
# ---------------------------------------------------------------------------

def output_grader(action: Action, task: TaskSpec) -> float:
    """
    Execute the fixed code and run each test case.

    Score = (number of passing tests) / (total tests).
    Returns 0.0 if the code raises an exception on any test.
    """
    if not task.test_inputs:
        return 0.5  # no tests → give partial credit

    # Safety gate: reject unsafe code
    try:
        from .safety import is_safe as _is_safe
        safe, _ = _is_safe(action.fixed_code)
    except Exception:
        safe = True
    if not safe:
        return 0.0

    lang = getattr(task, "language", "python")
    if lang == "javascript":
        return _js_output_grader(action, task)

    namespace, error = _run_code(textwrap.dedent(action.fixed_code))
    if error or namespace is None:
        return 0.0

    func = _resolve_entry_point(namespace, task)
    if func is None:
        return 0.0

    passed = 0
    for test in task.test_inputs:
        args = test.get("args", ())
        expected = test.get("expected")
        try:
            result = func(*args)
            if result == expected:
                passed += 1
        except Exception:
            pass  # counts as a failure

    return round(passed / len(task.test_inputs), 4)


def _resolve_entry_point(namespace: dict[str, Any], task: TaskSpec):
    """
    Find the function under test inside *namespace*.

    Prefers the task's declared ``entry_point`` so that agent-defined helpers
    cannot shadow or be mistaken for the target function.  Falls back to the
    first public callable only when no entry point is declared or present.
    """
    if task.entry_point:
        candidate = namespace.get(task.entry_point)
        if callable(candidate):
            return candidate
        return None

    # Backwards-compatible fallback for tasks without a declared entry point
    for name, obj in namespace.items():
        if callable(obj) and not name.startswith("_"):
            return obj
    return None


# ---------------------------------------------------------------------------
# Grader 3 — Semantic Similarity to Reference Solution
# ---------------------------------------------------------------------------

def similarity_grader(action: Action, task: TaskSpec) -> float:
    """
    Token-overlap similarity between the agent's fix and the reference fix.

    Uses a simple Jaccard coefficient over normalised code tokens.
    This rewards agents that adopt the canonical fix pattern.
    """
    def tokenise(code: str) -> set:
        # Split on whitespace and punctuation, lowercase, drop empties
        import re
        tokens = re.findall(r"[A-Za-z_]\w*|[0-9]+|[+\-*/=<>!:(),\[\]{}]", code)
        return set(t.lower() for t in tokens)

    agent_tokens = tokenise(action.fixed_code)
    ref_tokens = tokenise(task.fixed_code)

    if not agent_tokens and not ref_tokens:
        return 1.0
    if not agent_tokens or not ref_tokens:
        return 0.0

    intersection = len(agent_tokens & ref_tokens)
    union = len(agent_tokens | ref_tokens)
    return round(intersection / union, 4)


# ---------------------------------------------------------------------------
# Grader 4 — Explanation Quality
# ---------------------------------------------------------------------------

def explanation_grader(action: Action, task: TaskSpec) -> float:
    """
    Heuristic score for the quality of the agent's explanation.

    Checks length and presence of relevant keywords from the error/task.
    """
    explanation = action.explanation.strip()
    if not explanation:
        return 0.0

    word_count = len(explanation.split())
    # Length score: 0 words → 0.0, 20+ words → 0.5 max from length alone
    length_score = min(word_count / 40.0, 0.5)

    # Keyword relevance: does the explanation mention the error type or key terms?
    keywords = {
        "syntax": ["colon", "bracket", "indent", "syntax", "punctuation", "parenthes"],
        "logic": ["loop", "off-by-one", "condition", "index", "range", "logic", "bound"],
        "runtime": ["divide", "zero", "index", "type", "none", "empty", "error", "edge"],
    }
    relevant_words = keywords.get(task.error_type, [])
    lower_expl = explanation.lower()
    keyword_hits = sum(1 for w in relevant_words if w in lower_expl)
    keyword_score = min(keyword_hits / max(len(relevant_words), 1), 0.5)

    return round(length_score + keyword_score, 4)


# ---------------------------------------------------------------------------
# Grader 5 — Composite (used as the primary reward signal)
# ---------------------------------------------------------------------------

GRADER_WEIGHTS = {
    "syntax":      0.25,
    "output":      0.45,
    "similarity":  0.15,
    "explanation": 0.15,
}

def grade_components(action: Action, task: TaskSpec) -> dict[str, float]:
    """
    Run every individual grader once and return the component scores.

    Exposing the breakdown lets the environment distinguish *correctness*
    (syntax + output) from reward shaping, and lets callers report per-grader
    detail without re-running the (potentially expensive) output grader.
    """
    return {
        "syntax":      syntax_grader(action, task),
        "output":      output_grader(action, task),
        "similarity":  similarity_grader(action, task),
        "explanation": explanation_grader(action, task),
    }


def composite_from_components(scores: dict[str, float]) -> float:
    """Combine component scores into the weighted composite reward."""
    total = sum(GRADER_WEIGHTS[k] * scores.get(k, 0.0) for k in GRADER_WEIGHTS)
    return round(min(max(total, 0.0), 1.0), 4)


def composite_grader(action: Action, task: TaskSpec) -> float:
    """
    Weighted average of all four graders.  This is the primary reward signal.
    """
    return composite_from_components(grade_components(action, task))


# ---------------------------------------------------------------------------
# Grader registry
# ---------------------------------------------------------------------------

GRADER_REGISTRY = {
    "syntax":      syntax_grader,
    "output":      output_grader,
    "similarity":  similarity_grader,
    "explanation": explanation_grader,
    "composite":   composite_grader,
}
