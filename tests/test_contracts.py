"""Contract tests for environment state transitions."""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from openenv_env.environment import CodeDebugEnvironment  # noqa: E402
from openenv_env.models import Action  # noqa: E402
from openenv_env.tasks import TASK_REGISTRY  # noqa: E402


def test_reset_first_step_zero():
    env = CodeDebugEnvironment(random_seed=1)
    obs = env.reset(task_id="task_syntax_001")
    assert obs.step == 0
    assert not obs.done


def test_step_requires_reset():
    env = CodeDebugEnvironment()
    try:
        env.step(Action(fixed_code="def f(): pass", explanation="", confidence=1.0))
        assert False, "expected runtime error"
    except RuntimeError:
        return


def test_step_rejects_after_done():
    env = CodeDebugEnvironment()
    env.reset(task_id="task_syntax_001")
    a = TASK_REGISTRY["task_syntax_001"].fixed_code
    action = Action(fixed_code=a, explanation="", confidence=1.0)
    env.step(action)  # solved
    try:
        env.step(action)
        assert False, "expected runtime error"
    except RuntimeError:
        return


def test_step_count_increments():
    env = CodeDebugEnvironment()
    env.reset(task_id="task_syntax_001")
    a = "def multiply(a, b): return a * b"
    action = Action(fixed_code=a, explanation="", confidence=1.0)
    r = env.step(action)
    assert r.observation.step >= 1