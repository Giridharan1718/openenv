"""
Regression tests for the CodeDebug OpenEnv environment.

Each test pins one of the bugs fixed in the review:
  1. a correct fix must be marked solved (was unreachable via composite >= 0.9)
  2. the output grader must target the task's entry point, not the first callable
  3. concurrent clients must not share episode state
  4. the reported reward must be the best composite, never an inflated clamped sum
  5. inference.py must support --verbose and emit a [SUMMARY] line
"""

from __future__ import annotations

import ast
import glob
import importlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fastapi.testclient import TestClient  # noqa: E402

from openenv_env.environment import CodeDebugEnvironment  # noqa: E402
from openenv_env.graders import (  # noqa: E402
    composite_grader,
    explanation_grader,
    output_grader,
)
from openenv_env.models import Action  # noqa: E402
from openenv_env.server import create_app  # noqa: E402
from openenv_env.tasks import TASK_REGISTRY  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Correctness-based solved criterion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_id", list(TASK_REGISTRY))
def test_ground_truth_fix_is_solved(task_id: str) -> None:
    """Every ground-truth fix must be solved, even with no explanation."""
    task = TASK_REGISTRY[task_id]
    env = CodeDebugEnvironment()
    env.reset(task_id=task_id)

    result = env.step(Action(fixed_code=task.fixed_code, explanation=""))

    assert result.info["solved"] is True
    assert result.info["components"]["output"] == 1.0
    assert result.done is True


def test_incorrect_fix_is_not_solved() -> None:
    env = CodeDebugEnvironment()
    env.reset(task_id="task_logic_001")
    result = env.step(Action(fixed_code="def fibonacci(n):\n    return n\n"))
    assert result.info["solved"] is False


# ---------------------------------------------------------------------------
# 2. Entry-point targeting
# ---------------------------------------------------------------------------

def test_output_grader_ignores_helpers_defined_first() -> None:
    """A helper defined before the target must not be graded instead."""
    task = TASK_REGISTRY["task_syntax_001"]
    code = (
        "def helper():\n"
        "    return 99\n\n"
        "def multiply(a, b):\n"
        "    return a * b\n"
    )
    assert output_grader(Action(fixed_code=code), task) == 1.0


def test_output_grader_requires_the_entry_point() -> None:
    """Code that omits the target function scores 0, even if it defines others."""
    task = TASK_REGISTRY["task_syntax_001"]
    code = "def helper(a, b):\n    return a * b\n"
    assert output_grader(Action(fixed_code=code), task) == 0.0


# ---------------------------------------------------------------------------
# 3. Session isolation
# ---------------------------------------------------------------------------

def test_sessions_are_isolated() -> None:
    client = TestClient(create_app())
    client.post("/reset", json={"task_id": "task_syntax_001"}, headers={"X-Session-Id": "A"})
    client.post("/reset", json={"task_id": "task_logic_001"}, headers={"X-Session-Id": "B"})

    # Advance session A only.
    client.post(
        "/step",
        json={"action": {"fixed_code": "def multiply(a, b):\n    return a * b\n"}},
        headers={"X-Session-Id": "A"},
    )

    state_a = client.get("/state", headers={"X-Session-Id": "A"}).json()
    state_b = client.get("/state", headers={"X-Session-Id": "B"}).json()

    assert state_a["task_id"] == "task_syntax_001"
    assert state_a["step"] == 1
    # B must be completely untouched by A's activity.
    assert state_b["task_id"] == "task_logic_001"
    assert state_b["step"] == 0
    assert state_b["cumulative_reward"] == 0.0


def test_default_session_is_independent() -> None:
    client = TestClient(create_app())
    client.post("/reset", json={"task_id": "task_syntax_001"}, headers={"X-Session-Id": "A"})
    client.post("/reset", json={"task_id": "task_runtime_003"})  # default session

    default_state = client.get("/state").json()
    assert default_state["task_id"] == "task_runtime_003"


# ---------------------------------------------------------------------------
# 4. Reward semantics
# ---------------------------------------------------------------------------

def test_reward_is_best_not_inflated_sum() -> None:
    """Two imperfect steps must not sum to a perfect 1.0 (old clamp bug)."""
    env = CodeDebugEnvironment()
    env.reset(task_id="task_logic_001")

    r1 = env.step(Action(fixed_code="def fibonacci(n):\n    return n\n"))
    r2 = env.step(Action(fixed_code="def fibonacci(n):\n    if n <= 0:\n        return 0\n    return 1\n"))

    assert r1.done is False and r2.done is False  # neither solved
    state = env.state()
    assert 0.0 <= r1.reward <= 1.0 and 0.0 <= r2.reward <= 1.0
    # Reported reward is the best composite so far, not a clamped running sum.
    assert state.cumulative_reward == max(r1.reward, r2.reward)
    assert state.cumulative_reward < 1.0


def test_reward_stays_within_bounds_over_many_steps() -> None:
    env = CodeDebugEnvironment()
    env.reset(task_id="task_logic_001")
    for _ in range(TASK_REGISTRY["task_logic_001"].max_steps):
        result = env.step(Action(fixed_code="def fibonacci(n):\n    return n\n"))
        assert 0.0 <= result.reward <= 1.0
        assert 0.0 <= result.observation.reward <= 1.0


# ---------------------------------------------------------------------------
# 5. Grader calibration experiments quoted in the docs
# ---------------------------------------------------------------------------

def test_explanation_length_saturates_at_20_words() -> None:
    task = TASK_REGISTRY["task_syntax_001"]
    twenty = Action(fixed_code="x = 1", explanation=" ".join(["word"] * 20))
    assert explanation_grader(twenty, task) == pytest.approx(0.5)


def test_composite_bounds() -> None:
    for task_id, task in TASK_REGISTRY.items():
        score = composite_grader(Action(fixed_code=task.fixed_code, explanation=""), task)
        assert 0.0 <= score <= 1.0, task_id


# ---------------------------------------------------------------------------
# 6. inference.py CLI
# ---------------------------------------------------------------------------

def test_inference_verbose_and_summary() -> None:
    result = subprocess.run(
        [
            sys.executable,
            os.path.join(_ROOT, "inference.py"),
            "--tasks",
            "task_syntax_001",
            "--verbose",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert any(line.startswith("[STEP]") and "syntax=" in line for line in lines)
    assert any(line.startswith("[SUMMARY]") for line in lines)
    assert any(line.startswith("[END]") and "solved=true" in line for line in lines)


# ---------------------------------------------------------------------------
# 7. Packaging — the documented `pip install -e ".[dev]"` must work
# ---------------------------------------------------------------------------

def test_pyproject_build_backend_importable() -> None:
    """A non-existent build backend makes every documented install fail."""
    text = open(os.path.join(_ROOT, "pyproject.toml"), encoding="utf-8").read()
    match = re.search(r'^build-backend\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, "pyproject.toml declares no build-backend"
    backend = match.group(1).split(":")[0]
    importlib.import_module(backend)  # must be a real module


# ---------------------------------------------------------------------------
# 8. Course notebooks must run on the declared minimum Python (3.10)
# ---------------------------------------------------------------------------

#: Backslashes inside an f-string replacement field only became legal in 3.12
#: (PEP 701); the notebook kernel and requires-python are 3.10 / >=3.10.
_FSTRING_BACKSLASH = re.compile(r'f"[^"\n]*\\"')


@pytest.mark.parametrize("notebook", sorted(glob.glob(os.path.join(_ROOT, "course", "*", "*.ipynb"))))
def test_notebook_code_cells_run_on_minimum_python(notebook: str) -> None:
    cells = json.load(open(notebook, encoding="utf-8"))["cells"]
    code_cells = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    assert code_cells, f"{notebook} has no code cells"
    for src in code_cells:
        ast.parse(src)  # malformed cells would crash the lesson
        for line in src.splitlines():
            assert not _FSTRING_BACKSLASH.search(line), (
                f"{notebook}: backslash inside an f-string breaks Python < 3.12: {line.strip()}"
            )


# ---------------------------------------------------------------------------
# 9. Numbers quoted in the docs must match the code
# ---------------------------------------------------------------------------

def test_documented_step_example_numbers_are_accurate() -> None:
    """README/module-03 quote reward 0.905 and explanation 0.3667 for this payload."""
    payload = {
        "action": {
            "fixed_code": "def multiply(a, b):\n    result = a * b\n    return result\n",
            "explanation": "Added the missing colon after the function signature.",
            "confidence": 0.95,
        }
    }
    client = TestClient(create_app())
    client.post("/reset", json={"task_id": "task_syntax_001"})
    body = client.post("/step", json=payload).json()

    assert body["reward"] == 0.905
    assert body["info"]["components"]["explanation"] == 0.3667
    assert body["done"] is True and body["info"]["solved"] is True

    for name in ("README.md", os.path.join("course", "module_03_inference", "README.md")):
        text = open(os.path.join(_ROOT, name), encoding="utf-8").read()
        assert "[SUMMARY]" in text, f"{name} omits the [SUMMARY] line"
    readme = open(os.path.join(_ROOT, "README.md"), encoding="utf-8").read()
    assert "0.905" in readme and "0.3667" in readme


# ---------------------------------------------------------------------------
# 10. CI covers the check that cannot be run everywhere: the Docker image
# ---------------------------------------------------------------------------

_SMOKE = os.path.join(_ROOT, "scripts", "smoke_test.py")
_WORKFLOW = os.path.join(_ROOT, ".github", "workflows", "ci.yml")


def _free_port() -> int:
    """A port that was free a moment ago (nothing is listening on it)."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_smoke_test_is_stdlib_only() -> None:
    """CI runs it with bare system python, before project deps are installed."""
    tree = ast.parse(open(_SMOKE, encoding="utf-8").read())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    third_party = sorted(m for m in imported if m not in sys.stdlib_module_names)
    assert not third_party, f"smoke_test.py must stay stdlib-only, found: {third_party}"


def test_smoke_test_fails_loudly_when_nothing_is_listening() -> None:
    """A smoke test that cannot fail is not a gate."""
    port = _free_port()
    result = subprocess.run(
        [sys.executable, _SMOKE, "--base-url", f"http://127.0.0.1:{port}", "--timeout", "2"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1, result.stdout
    assert "[FAIL]" in result.stdout and "health" in result.stdout
    assert "0/1 checks passed" in result.stdout


def test_smoke_test_passes_against_a_real_server() -> None:
    """Boots app.py and exercises it over HTTP exactly as the CI job does."""
    port = _free_port()
    env = dict(os.environ, PORT=str(port), HOST="127.0.0.1", PYTHONUNBUFFERED="1")
    server = subprocess.Popen(
        [sys.executable, os.path.join(_ROOT, "app.py")],
        cwd=_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        health = f"http://127.0.0.1:{port}/api/v1/health"
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                urllib.request.urlopen(health, timeout=2).close()
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise AssertionError(f"app.py never answered on port {port}")

        result = subprocess.run(
            [sys.executable, _SMOKE, "--base-url", f"http://127.0.0.1:{port}", "--timeout", "30"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "4/4 checks passed" in result.stdout
    finally:
        server.terminate()
        try:
            server.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()


def test_ci_workflow_builds_the_image_and_smoke_tests_health() -> None:
    """The workflow must actually gate on the image building and answering."""
    import yaml  # dev dependency; kept local so the rest of the suite runs bare

    assert os.path.exists(_WORKFLOW), "CI workflow is missing"
    doc = yaml.safe_load(open(_WORKFLOW, encoding="utf-8"))
    triggers = doc.get("on", doc.get(True))
    assert {"push", "pull_request"} <= set(triggers)

    jobs = doc["jobs"]
    assert "docker" in jobs, "CI must build the image; that is the gap it closes"
    docker_steps = jobs["docker"]["steps"]
    scripts = "\n".join(step.get("run", "") for step in docker_steps)
    assert "docker build" in scripts
    assert "scripts/smoke_test.py" in scripts, "CI must smoke-test the served API"
    assert "docker rm -f" in scripts, "failed runs must not leak containers"

    for job in jobs.values():
        for step in job["steps"]:
            assert not step.get("continue-on-error"), (
                f"CI step '{step.get('name')}' is allowed to fail, so it proves nothing"
            )
