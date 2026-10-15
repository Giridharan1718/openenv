"""Run the reproducible CodeDebug benchmark."""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from openenv_env.evaluator import run_benchmark  # noqa: E402
from openenv_env.models import Action  # noqa: E402
from openenv_env.tasks import TASK_REGISTRY  # noqa: E402


def main() -> None:
    def solver(obs, tid):
        task = TASK_REGISTRY[tid]
        return Action(fixed_code=task.fixed_code, explanation="", confidence=1.0)

    res = run_benchmark(solver)
    print(json.dumps(res["summary"], indent=2))


if __name__ == "__main__":
    main()
