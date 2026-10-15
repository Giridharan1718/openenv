#!/usr/bin/env python
"""Check for multi-language runtimes required by the executors."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from openenv_env.language_executors import REGISTRY  # noqa: E402

INSTALL_HINTS = {
    "python": "Usually preinstalled; see https://python.org",
    "javascript": "Install Node.js: https://nodejs.org (must be on PATH for Python subprocess)",
    "typescript": "npm install -g typescript",
    "go": "https://go.dev/dl/ (add go.exe/go to PATH)",
    "rust": "https://rustup.rs/ (rustc/cargo on PATH)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="exit non-zero if any missing")
    args = ap.parse_args()

    missing = []
    for lang, exe in REGISTRY.items():
        avail = bool(getattr(exe, "available", lambda: False)())
        print(f"{lang:12s} {'OK' if avail else 'MISSING'}")
        if not avail:
            missing.append(lang)
            print(f"    install: {INSTALL_HINTS.get(lang)}")

    if args.strict and missing:
        print(f"\nMissing runtimes: {', '.join(missing)}")
        raise SystemExit(1)
    print(f"\n{len(REGISTRY)-len(missing)}/{len(REGISTRY)} runtimes available")


if __name__ == "__main__":
    main()