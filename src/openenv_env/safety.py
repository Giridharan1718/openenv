"""safety.py — static safety filter for submitted code/tasks.

Rejects code that uses dangerous operations: shell-out, exec/eval, network
access, file writes outside sandbox, privilege escalation, resource abuse.
"""
from __future__ import annotations

import ast
import re

_DANGEROUS_CALLS = {
    "exec", "eval", "compile", "__import__",
    "os.system", "os.popen", "subprocess.run", "subprocess.call",
    "subprocess.Popen", "subprocess.check_output",
    "requests.get", "urllib.request.urlopen", "socket.socket",
}

_DANGEROUS_IMPORTS = {
    "os", "subprocess", "socket", "requests", "urllib", "ctypes", "pickle",
    "shutil", "sys",
}

_DANGEROUS_ATTRS = {"system", "popen", "remove", "rmdir", "unlink", "chmod", "chown"}


def is_safe(source: str) -> tuple[bool, list[str]]:
    reasons = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, [f"syntax error: {e}"]
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else [node.module.split(".")[0] if node.module else ""]
            for n in names:
                if n in _DANGEROUS_IMPORTS:
                    reasons.append(f"disallowed import: {n}")
        if isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = f"{getattr(func.value, 'id', '')}.{func.attr}"
            if name in _DANGEROUS_CALLS or func.attr in _DANGEROUS_ATTRS if isinstance(func, ast.Attribute) else False:
                reasons.append(f"disallowed call: {name}")
        if isinstance(node, ast.Attribute) and node.attr in _DANGEROUS_ATTRS:
            reasons.append(f"dangerous attribute: {node.attr}")
    # regex backstop
    for pat in (r"\bopen\s*\(",):
        if re.search(pat, source):
            reasons.append("file open detected")
    return (len(reasons) == 0), reasons