"""
environment.py — CodeDebug OpenEnv-compliant Environment.

Implements the three lifecycle methods expected by OpenEnv:
  reset()  → Observation
  step()   → StepResult
  state()  → EnvironmentState
"""

from __future__ import annotations

import random
import time
from typing import Any

from .graders import composite_from_components, grade_components
from .models import (
    Action,
    DifficultyLevel,
    EnvironmentState,
    ErrorType,
    Observation,
    StepResult,
)
from .tasks import DEFAULT_TASK_ORDER, TASK_REGISTRY, TaskSpec


class CodeDebugEnvironment:
    """
    A multi-task code-debugging training environment.

    Episode flow
    ------------
    1. reset(task_id?)  — start a fresh episode on a given (or random) task
    2. step(action)     — agent submits a fix, environment scores it
    3. state()          — read-only snapshot at any time

    The episode ends when:
      * the submitted fix is *correct* — it parses and passes every test case
        (``solved``), or
      * the step limit (task.max_steps) is reached.

    Note on the reward signal
    -------------------------
    Every step returns the weighted composite score, which mixes correctness
    (syntax + output) with shaping terms (similarity + explanation).  Whether
    the episode is *solved* is judged purely on correctness so that phrasing a
    perfect fix in a different way can never prevent it from being recognised.
    The reward reported on the observation is the best (maximum) composite
    score seen so far, so it stays within [0, 1] and is never inflated by
    summing multiple steps.
    """

    #: Output grader score that marks an episode as solved (all tests pass).
    SOLVE_OUTPUT_THRESHOLD = 1.0

    def __init__(
        self,
        task_order: list[str] | None = None,
        random_seed: int | None = None,
    ) -> None:
        self._task_order = task_order or DEFAULT_TASK_ORDER
        self._rng = random.Random(random_seed)

        # Episode state
        self._task: TaskSpec | None = None
        self._step: int = 0
        self._done: bool = False
        self._best_reward: float = 0.0
        self._current_observation: Observation | None = None
        self._history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, task_id: str | None = None) -> Observation:
        """
        Start a new episode.

        Parameters
        ----------
        task_id : str, optional
            Pick a specific task.  If None a random task is selected.

        Returns
        -------
        Observation
            The initial observation for the new episode.
        """
        if task_id and task_id in TASK_REGISTRY:
            self._task = TASK_REGISTRY[task_id]
        else:
            chosen_id = self._rng.choice(self._task_order)
            self._task = TASK_REGISTRY[chosen_id]

        self._step = 0
        self._done = False
        self._best_reward = 0.0
        self._history = []

        obs = self._build_observation()
        self._current_observation = obs
        return obs

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: Action) -> StepResult:
        """
        Submit an action and advance the environment by one step.

        Parameters
        ----------
        action : Action
            The agent's proposed fix.

        Returns
        -------
        StepResult
            Contains the next Observation, the step reward, done flag, and info.

        Raises
        ------
        RuntimeError
            If called before reset() or after the episode has ended.
        """
        if self._task is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        # Score the action once, keeping the per-grader breakdown
        components = grade_components(action, self._task)
        reward = composite_from_components(components)
        self._step += 1
        self._best_reward = max(self._best_reward, reward)

        # A fix is solved when it parses and passes every test case.
        correct = (
            components["syntax"] >= 1.0
            and components["output"] >= self.SOLVE_OUTPUT_THRESHOLD
        )
        timeout = self._step >= self._task.max_steps
        self._done = correct or timeout

        # Build next observation
        obs = self._build_observation()
        self._current_observation = obs

        # Record history entry
        self._history.append({
            "step": self._step,
            "action_summary": action.fixed_code[:80],
            "reward": reward,
            "done": self._done,
            "timestamp": time.time(),
        })

        info = {
            "solved": correct,
            "timeout": timeout,
            "step_reward": reward,
            "best_reward": self._best_reward,
            # Retained for backwards compatibility; equals the best score so far.
            "cumulative_reward": self._best_reward,
            "components": components,
        }

        return StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info=info,
        )

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def state(self) -> EnvironmentState:
        """Return a read-only snapshot of the current environment state."""
        return EnvironmentState(
            task_id=self._task.task_id if self._task else None,
            step=self._step,
            done=self._done,
            cumulative_reward=self._best_reward,
            current_observation=self._current_observation,
            history=list(self._history),
        )

    def close(self) -> None:
        """Release any environment resources.

        The current implementation is in-process and does not hold external
        resources, so this is intentionally a no-op.
        """
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        """Construct an Observation from current episode state."""
        assert self._task is not None

        # Reveal hints progressively as the step count increases
        hints_to_show = self._task.hints[: self._step]

        return Observation(
            task_id=self._task.task_id,
            step=self._step,
            buggy_code=self._task.buggy_code,
            error_message=self._task.error_message,
            error_type=ErrorType(self._task.error_type),
            difficulty=DifficultyLevel(self._task.difficulty),
            max_steps=self._task.max_steps,
            hints=hints_to_show,
            done=self._done,
            reward=self._best_reward,
            metadata={"description": self._task.description},
        )
