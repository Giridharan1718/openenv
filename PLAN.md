# OpenEnv — Enhancement Roadmap

Implements ideas 1-10 as surveyed. Each phase is independently shippable.

## Phase 0 — Foundations (DONE)
- Multi-language `language` field on TaskSpec
- `_js_output_grader` via node (JS tasks shipped)
- ruff clean, 28/28 tests, validation 54/54

## Phase 1 — Language executors (JS done; TS/Go/Rust)
- Add `language ∈ {python, javascript, typescript, go, rust}`.
- Per-language sandbox executor in `graders.py`:
  - TS: `tsc` compile → node, or `tsx`
  - Go: `go run` in temp module
  - Rust: `rustc` + run binary
- Add 2 tasks × 4 langs (syntax + logic) ≈ 8 tasks.
- Manifest: surface `language`.
- Acceptance: all task graders return sane scores; `validate_submission` +72 checks.

## Phase 2 — Multi-turn reasoning + patching
- Extend `TaskSpec` with `max_steps`, `hint_schedule`, `partial_test_reveal`.
- Environment lets agent: request hint, run partial tests, submit patch (unified diff).
- `Action.fixed_code` becomes optional; new `Action.patch` applied to buggy code.
- Reward shaping: incremental over steps; composite is best-so-far.
- Acceptance: episode length > 1 useful; tests no regression.

## Phase 3 — Agentic debugging (tools)
- Define tool schema: `search_web`, `run_tests`, `view_file`, `patch_code`.
- Agent loop in `inference.py`: plan → tool call → observe → next step.
- Tool executor sandboxed; limits (max calls, timeout).
- Acceptance: agent solves ≥1 more task than one-shot; logs structured.

## Phase 4 — Auto-generate tasks
- LLM generator: prompt → buggy code + tests + reference fix.
- Validator: ensure tests fail on buggy, pass on reference; seed validity gate.
- CLI: `python scripts/generate_tasks.py --n 20 --lang python`.
- Acceptance: ≥80% validity; tasks land in `tasks/generated/`.

## Phase 5 — Real-world issues
- Scraper: GitHub Issues/PRs labeled "good first issue" → task candidates.
- License filter; PII redaction.
- Frozen evaluation set; no overlap with train.
- Acceptance: 50 real-world tasks curated.

## Phase 6 — Benchmarks & leaderboard
- HF Spaces eval harness (`/eval` endpoint, SSE progress).
- Public leaderboard JSON store; submit endpoint with token.
- Docker pinned; reproducibility (seed + version).
- Acceptance: one-click run from Space; downloadable scores.

## Phase 7 — IDE integration (Cursor-like)
- VS Code extension: commands "Debug with OpenEnv", live diagnostics.
- LSP-ish bridge to local env server; streams observations.
- Acceptance: manual demo working.

## Phase 8 — Safety filter
- Static analyzer: reject `os.system`, `eval`, network, file writes.
- Allow-list APIs; sandbox exec with resource limits (cpu/mem/time).
- Audit log of attempted malicious submissions.
- Acceptance: known bad snippets blocked; benign pass.

## Cross-cutting
- CI: run pytest + ruff + validate on PR; multi-lang smoke (where available)
- Docs: course module per new feature.
- Metrics: per-task solve rate, steps-to-solve distribution.

## Effort estimate (days, one dev)
P1 3d · P2 3d · P3 4d · P4 2d · P5 4d · P6 3d · P7 5d · P8 2d ≈ 26 days.
"