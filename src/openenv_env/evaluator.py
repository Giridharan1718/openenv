"""evaluator.py — reproducible benchmark harness for CodeDebug."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Callable

from .environment import CodeDebugEnvironment
from .models import Action
from .tasks import TASK_REGISTRY


@dataclass
class EvalResult:
    task_id: str
    language: str
    difficulty: str
    solved: bool
    attempts: int
    best_reward: float
    elapsed_s: float


def run_benchmark(solver: Callable[[Any, str], Action], task_ids: list[str] | None = None) -> dict[str, Any]:
    env = CodeDebugEnvironment()
    results: list[EvalResult] = []
    for tid in (task_ids or list(TASK_REGISTRY.keys())):
        task = TASK_REGISTRY[tid]
        t0 = time.time()
        obs = env.reset(task_id=tid)
        action = solver(obs, tid)
        step = env.step(action)
        results.append(EvalResult(
            task_id=tid,
            language=getattr(task, "language", "python"),
            difficulty=task.difficulty,
            solved=bool(step.info.get("solved", False)),
            attempts=1,
            best_reward=step.info.get("best_reward", step.reward),
            elapsed_s=round(time.time() - t0, 4),
        ))
    total = len(results)
    solved = sum(1 for r in results if r.solved)
    return {
        "tasks": [asdict(r) for r in results],
        "summary": {
            "total": total,
            "solved": solved,
            "success_rate": round(solved / total, 4) if total else 0.0,
        },
    }