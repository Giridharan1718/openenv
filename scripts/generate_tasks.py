#!/usr/bin/env python
"""Auto-generate debugging tasks via an OpenAI-compatible LLM (optional API key).

Usage (dry run without API):
    python scripts/generate_tasks.py --dry-run --n 3
Usage (real):
    API_BASE_URL=... MODEL_NAME=... HF_TOKEN=... python scripts/generate_tasks.py --n 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))


def dry_run_task(i: int) -> dict:
    return {
        "task_id": f"task_gen_{i:03d}",
        "language": "python",
        "description": "Generated: fix an off-by-one in a sum function.",
        "entry_point": "total",
        "buggy_code": "def total(nums):\n    s = 0\n    for i in range(len(nums) - 1):  # BUG\n        s += nums[i]\n    return s\n",
        "fixed_code": "def total(nums):\n    s = 0\n    for i in range(len(nums)):\n        s += nums[i]\n    return s\n",
        "error_type": "logic",
        "difficulty": "easy",
        "test_inputs": [{"args": ([1, 2, 3],), "expected": 6}],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="tasks/generated")
    args = ap.parse_args()

    tasks = []
    for i in range(args.n):
        if args.dry_run:
            tasks.append(dry_run_task(i))
        else:
            # Real generation would call OpenAI here; placeholder.
            tasks.append(dry_run_task(i))

    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    out_path = os.path.join(ROOT, args.out, "generated.json")
    with open(out_path, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"Wrote {len(tasks)} tasks to {out_path}")


if __name__ == "__main__":
    main()