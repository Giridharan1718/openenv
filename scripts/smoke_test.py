#!/usr/bin/env python3
"""Smoke-test a *running* CodeDebug OpenEnv server over HTTP.

This is the check CI runs against the built Docker image, because the
in-process tests cannot prove that the container actually serves traffic:

    docker build -t codedebug-env .
    docker run -d --name codedebug -p 7860:7860 codedebug-env
    python scripts/smoke_test.py --base-url http://127.0.0.1:7860

It is stdlib-only on purpose so it needs no extra installs and can even run
inside the image.  Exit status is 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def _request(
    url: str,
    method: str = "GET",
    payload: Any = None,
    timeout: float = 10.0,
) -> tuple[int, str]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode()


def wait_for_health(base_url: str, timeout: float) -> str:
    """Poll /api/v1/health until it returns 200, bounded by ``timeout`` seconds.

    A container needs a moment to bind its port, so this is the only check with
    retries; everything after it must pass immediately or the run fails.
    """
    deadline = time.monotonic() + timeout
    last = "no response yet"
    while time.monotonic() < deadline:
        try:
            status, body = _request(f"{base_url}/api/v1/health", timeout=5.0)
            if status == 200:
                return body
            last = f"HTTP {status}"
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(1)
    raise TimeoutError(
        f"{base_url}/api/v1/health never returned 200 within {timeout:.0f}s (last: {last})"
    )


def _json(body: str) -> Any:
    return json.loads(body)


def run_checks(base_url: str, timeout: float) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    # 1. Health — proves the server booted and is reachable from outside.
    try:
        body = wait_for_health(base_url, timeout)
        health = _json(body)
        ok = isinstance(health, dict) and "status" in health
        results.append(
            ("GET /api/v1/health returns 200 with 'status'", ok, body.strip()[:120])
        )
    except Exception as exc:  # health is the gate: nothing else can work
        results.append(("GET /api/v1/health returns 200 with 'status'", False, str(exc)))
        return results

    # 2. Reset — starts an episode and returns an Observation.
    try:
        status, body = _request(f"{base_url}/reset", method="POST", payload={})
        obs = _json(body)
        wanted = ("task_id", "buggy_code", "error_type", "step")
        missing = [k for k in wanted if k not in obs]
        results.append(
            (
                "POST /reset returns an Observation",
                status == 200 and not missing,
                f"status={status} task_id={obs.get('task_id')}"
                + (f" missing={missing}" if missing else ""),
            )
        )
    except Exception as exc:
        results.append(("POST /reset returns an Observation", False, str(exc)))

    # 3. State — the environment must be readable between steps.
    try:
        status, body = _request(f"{base_url}/state")
        results.append(("GET /state returns 200", status == 200, f"status={status}"))
    except Exception as exc:
        results.append(("GET /state returns 200", False, str(exc)))

    # 4. Step — the reward path must actually score an action.
    try:
        action = {
            "action": {
                "fixed_code": "def multiply(a, b):\n    return a * b\n",
                "explanation": "Added the missing colon.",
                "confidence": 0.9,
            }
        }
        status, body = _request(f"{base_url}/step", method="POST", payload=action)
        step = _json(body)
        reward = step.get("reward")
        keys_ok = all(k in step for k in ("observation", "reward", "done"))
        bound_ok = isinstance(reward, (int, float)) and 0.0 <= reward <= 1.0
        results.append(
            (
                "POST /step returns StepResult with reward in [0,1]",
                status == 200 and keys_ok and bound_ok,
                f"status={status} reward={reward}",
            )
        )
    except Exception as exc:
        results.append(("POST /step returns StepResult with reward in [0,1]", False, str(exc)))

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-test a running CodeDebug server.")
    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:7860",
        help="Origin of the running server (default: %(default)s)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for the health endpoint (default: %(default)s)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    print(f"Smoke-testing {base_url} ...")
    results = run_checks(base_url, args.timeout)

    failed = 0
    for name, ok, detail in results:
        icon = "PASS" if ok else "FAIL"
        print(f"  [{icon}] {name}" + (f"\n         {detail}" if detail else ""))
        if not ok:
            failed += 1

    total = len(results)
    print(f"  RESULT: {total - failed}/{total} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
