"""language_executors.py — modular language execution adapters.

Each adapter exposes:
    syntax_check(code, timeout) -> (ok, error)
    run(code, entry_point, tests, timeout) -> list of result dicts
and reports `available()`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, Callable

DEFAULT_TIMEOUT = 10


def _which(names: list[str]) -> str | None:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


class LanguageExecutor:
    language: str = "python"

    def available(self) -> bool:
        return True

    def syntax_check(self, code: str, timeout: int = DEFAULT_TIMEOUT) -> tuple[bool, str]:
        raise NotImplementedError

    def run_tests(self, code: str, entry_point: str | None, tests: list[dict],
                  timeout: int = DEFAULT_TIMEOUT) -> list[dict]:
        raise NotImplementedError


class PythonExecutor(LanguageExecutor):
    language = "python"

    def syntax_check(self, code, timeout=DEFAULT_TIMEOUT):
        try:
            compile(textwrap.dedent(code), "<agent>", "exec")
            return True, ""
        except SyntaxError as e:
            return False, str(e)

    def run_tests(self, code, entry_point, tests, timeout=DEFAULT_TIMEOUT):
        results = []
        namespace: dict[str, Any] = {}
        try:
            exec(compile(textwrap.dedent(code), "<agent>", "exec"), namespace)  # noqa: S102
        except Exception as e:
            return [{"passed": False, "error": repr(e)} for _ in tests] or [{"passed": False, "error": repr(e)}]
        func = namespace.get(entry_point) if entry_point else None
        if not callable(func):
            return [{"passed": False, "error": "entry point not found"} for _ in tests]
        for t in tests:
            args = t.get("args", ())
            expected = t.get("expected")
            try:
                got = func(*args)
                results.append({"passed": got == expected, "actual": got, "expected": expected})
            except Exception as e:
                results.append({"passed": False, "error": repr(e), "expected": expected})
        return results


class JavaScriptExecutor(LanguageExecutor):
    language = "javascript"

    def __init__(self):
        self._node = _which(["node"]) or _which(["node.exe"])

    def available(self):
        # spawn test
        if not self._node:
            return False
        try:
            r = subprocess.run([self._node, "--version"], capture_output=True, text=True, timeout=5)
            return r.returncode == 0
        except Exception:
            return False

    def _write(self, content, suffix):
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "w") as f:
            f.write(content)
        return path

    def syntax_check(self, code, timeout=DEFAULT_TIMEOUT):
        if not self.available():
            return False, "node runtime unavailable"
        p = self._write(textwrap.dedent(code), ".mjs")
        try:
            r = subprocess.run([self._node, "--check", p], capture_output=True, text=True, timeout=timeout)
            return (r.returncode == 0, (r.stderr or "").strip())
        finally:
            try:
                os.unlink(p)
            except OSError:
                pass

    def run_tests(self, code, entry_point, tests, timeout=DEFAULT_TIMEOUT):
        if not self.available():
            return [{"passed": False, "error": "node runtime unavailable"} for _ in tests]
        mod = self._write(textwrap.dedent(code), ".mjs")
        out = []
        try:
            for t in tests:
                args = t.get("args", [])
                expected = t.get("expected")
                script = (
                    f"import {{ {entry_point} }} from 'file:///{mod.replace(chr(92), '/')}';\n"
                    f"const args = JSON.parse(process.argv[1]);\n"
                    f"let r; try {{ r = {entry_point}(...args); }} catch(e) {{ console.error(e && e.message); process.exit(2); }}\n"
                    f"console.log(JSON.stringify(r));\n"
                )
                sf = self._write(script, ".mjs")
                try:
                    r = subprocess.run([self._node, sf, json.dumps(list(args))], capture_output=True, text=True, timeout=timeout)
                except subprocess.TimeoutExpired:
                    out.append({"passed": False, "error": "timeout", "expected": expected})
                    continue
                finally:
                    try:
                        os.unlink(sf)
                    except OSError:
                        pass
                if r.returncode != 0:
                    out.append({"passed": False, "error": (r.stderr or "runtime error").strip(), "expected": expected})
                    continue
                try:
                    got = json.loads(r.stdout.strip())
                except Exception:
                    out.append({"passed": False, "error": "bad output", "expected": expected})
                    continue
                out.append({"passed": got == expected, "actual": got, "expected": expected})
        finally:
            try:
                os.unlink(mod)
            except OSError:
                pass
        return out


class TypescriptExecutor(JavaScriptExecutor):
    language = "typescript"

    def __init__(self):
        super().__init__()
        self._tsc = _which(["tsc"])

    def available(self):
        return super().available() and bool(self._tsc)

    def syntax_check(self, code, timeout=DEFAULT_TIMEOUT):
        if not self.available():
            return False, "tsc runtime unavailable"
        # Delegate to tsc via temp project is complex; fall back to node check on JS-stripped? Minimal: report unavailable-feature.
        return False, "typescript adapter not fully implemented in this environment"


class GoExecutor(LanguageExecutor):
    language = "go"

    def __init__(self):
        self._go = _which(["go"])

    def available(self):
        return bool(self._go)

    def syntax_check(self, code, timeout=DEFAULT_TIMEOUT):
        if not self.available():
            return False, "go runtime unavailable"
        return False, "go adapter not fully implemented in this environment"

    def run_tests(self, code, entry_point, tests, timeout=DEFAULT_TIMEOUT):
        return [{"passed": False, "error": "go runtime unavailable or not implemented"} for _ in tests]


class RustExecutor(LanguageExecutor):
    language = "rust"

    def __init__(self):
        self._rustc = _which(["rustc", "cargo"])

    def available(self):
        return bool(self._rustc)

    def syntax_check(self, code, timeout=DEFAULT_TIMEOUT):
        if not self.available():
            return False, "rust runtime unavailable"
        return False, "rust adapter not fully implemented in this environment"

    def run_tests(self, code, entry_point, tests, timeout=DEFAULT_TIMEOUT):
        return [{"passed": False, "error": "rust runtime unavailable or not implemented"} for _ in tests]


REGISTRY: dict[str, LanguageExecutor] = {
    "python": PythonExecutor(),
    "javascript": JavaScriptExecutor(),
    "typescript": TypescriptExecutor(),
    "go": GoExecutor(),
    "rust": RustExecutor(),
}


def get_executor(language: str) -> LanguageExecutor:
    return REGISTRY.get(language, REGISTRY["python"])
PYEOF