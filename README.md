---
title: OpenEnv CodeDebug Environment
emoji: "🐛"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
license: mit
tags:
  - openenv
  - code-debugging
  - fastapi
  - python
language:
  - en
---

# 🐛 CodeDebug — OpenEnv Hackathon Environment

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-orange)](https://huggingface.co/openenv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A fully **Meta PyTorch OpenEnv-compliant** training environment where LLM agents
learn to debug Python code.  Covers **syntax errors**, **logic errors**, and
**runtime errors** across easy / medium / hard difficulty levels.

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Local Run](#local-run)
4. [Docker Build & Run](#docker-build--run)
5. [OpenEnv Endpoint Usage](#openenv-endpoint-usage)
6. [Custom API v1 Examples](#custom-api-v1-examples)
7. [Inference Setup](#inference-setup)
8. [Validator Commands](#validator-commands)
9. [Continuous Integration](#continuous-integration)
10. [Hugging Face Spaces Deployment](#hugging-face-spaces-deployment)
11. [Pre-Submission Checklist](#pre-submission-checklist)
12. [Course Content](#course-content)

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/<you>/codedebug-env
cd codedebug-env

# 2. Install
pip install -r requirements.txt

# 3. Run the server
python app.py
# → http://localhost:7860/docs

# 4. Smoke-test
curl http://localhost:7860/api/v1/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d "{}"
```

---

## Project Structure

```
codedebug-env/
├── src/
│   └── openenv_env/
│       ├── __init__.py          # public API
│       ├── models.py            # Pydantic Action / Observation / StepResult
│       ├── tasks.py             # TaskSpec registry (7 tasks)
│       ├── graders.py           # 5 graders; composite reward in [0,1]
│       ├── environment.py       # CodeDebugEnvironment (reset/step/state)
│       └── server.py            # FastAPI app via create_app()
├── course/
│   ├── index.md
│   ├── module_01_intro/         README.md + notebook.ipynb
│   ├── module_02_environment/   README.md + notebook.ipynb
│   ├── module_03_inference/     README.md + notebook.ipynb
│   ├── module_04_graders/       README.md + notebook.ipynb
│   └── module_05_deployment/    README.md + notebook.ipynb
├── tests/
│   └── test_codedebug.py        # Regression suite (pytest)
├── scripts/
│   └── smoke_test.py            # HTTP smoke test for a running server
├── .github/workflows/ci.yml     # CI: validate + Docker build + health smoke test
├── inference.py                 # LLM agent evaluation harness
├── validate_submission.py       # Pre-submission validator
├── app.py                       # Docker entrypoint
├── openenv.yaml                 # OpenEnv metadata/spec file
├── manifest.json                # OpenEnv manifest
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Local Run

### Development server (with auto-reload)
```bash
pip install -e ".[dev]"
uvicorn openenv_env.server:create_app --factory --port 8000 --reload
```

### Production server
```bash
python app.py                        # uses PORT env var (default 7860)
PORT=8080 python app.py              # custom port
```

### Interactive API docs
Open http://localhost:7860/docs (Swagger UI) or http://localhost:7860/redoc

---

## Docker Build & Run

```bash
# Build
docker build -t codedebug-env:latest .

# Run
docker run -p 7860:7860 codedebug-env:latest

# Run with inference env vars
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=hf_your_token_here \
  codedebug-env:latest

# Confirm it is up
curl http://localhost:7860/api/v1/health
```

---

## OpenEnv Endpoint Usage

### POST /reset
Start a new debugging episode (random task):
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{}'
```

Start a specific task:
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_syntax_001"}'
```

Example response:
```json
{
  "task_id": "task_syntax_001",
  "step": 0,
  "buggy_code": "def multiply(a, b)\n    result = a * b\n    return result\n",
  "error_message": "SyntaxError: expected ':'",
  "error_type": "syntax",
  "difficulty": "easy",
  "max_steps": 3,
  "hints": [],
  "done": false,
  "reward": 0.0,
  "metadata": {"description": "Fix a missing colon in a function definition."}
}
```

### POST /step
Submit the agent's fix:
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "fixed_code": "def multiply(a, b):\n    result = a * b\n    return result\n",
      "explanation": "Added the missing colon after the function signature.",
      "confidence": 0.95
    }
  }'
```

Example response:
```json
{
  "observation": {"task_id": "task_syntax_001", "step": 1, "done": true, ...},
  "reward": 0.905,
  "done": true,
  "info": {
    "solved": true,
    "timeout": false,
    "step_reward": 0.905,
    "best_reward": 0.905,
    "cumulative_reward": 0.905,
    "components": {"syntax": 1.0, "output": 1.0, "similarity": 1.0, "explanation": 0.3667}
  }
}
```

### GET /state
Read the current environment state at any time:
```bash
curl http://localhost:7860/state
```

---

## Custom API v1 Examples

### Health check
```bash
curl http://localhost:7860/api/v1/health
# {"status": "ok", "uptime_seconds": 12.4, "version": "1.0.0", ...}
```

### Versioned reset / step / state
```bash
# Reset
curl -X POST http://localhost:7860/api/v1/reset -H "Content-Type: application/json" -d '{}'

# Step
curl -X POST http://localhost:7860/api/v1/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"fixed_code": "def f(x):\n    return x\n", "explanation": "test"}}'

# State
curl http://localhost:7860/api/v1/state
```

---

## Inference Setup

### Environment Variables

| Variable       | Required | Description | Example |
|----------------|----------|-------------|---------|
| `API_BASE_URL` | Yes      | OpenAI-compatible base URL | `https://api.openai.com/v1` |
| `MODEL_NAME`   | Yes      | Model identifier | `gpt-4o-mini` |
| `HF_TOKEN`     | Optional | HF token (used as API key for HF Inference) | `hf_xxx...` |

### Run inference

```bash
# Set env vars
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=sk-your-key-here   # or HF_TOKEN

# Run all default tasks (first 3)
python inference.py

# Run specific tasks
python inference.py --tasks task_syntax_001 task_logic_001 task_runtime_001

# With verbose output + save results
python inference.py --tasks task_syntax_001 --verbose --output results.json

# Override max steps (useful for quick tests)
python inference.py --max-steps 1

# Using Hugging Face Inference Endpoint
export API_BASE_URL=https://api-inference.huggingface.co/models/meta-llama/Llama-3.3-70B-Instruct/v1
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export HF_TOKEN=hf_your_token_here
python inference.py
```

### Structured log format
Every line on stdout follows the fixed text format expected by the benchmark.
Real output for `python inference.py --tasks task_syntax_001`:
```text
[START] task=task_syntax_001 env=codedebug-env model=gpt-4o-mini
[STEP] step=1 action=def multiply(a, b):\n    result = a * b\n    return result\n reward=0.91 done=true error=null
[END] task=task_syntax_001 success=true solved=true steps=1 score=0.91 rewards=0.91
[SUMMARY] total=1 solved=1 avg_score=0.9050 elapsed=0.0s within_limit=true
```

The validator scans for lines that **start with** `[START]`, `[STEP]`, or `[END]`;
`[SUMMARY]` is an additional one-line roll-up printed last.  Add `--verbose` to
append the per-grader breakdown to every `[STEP]` line:

```text
[STEP] step=1 action=... reward=0.91 done=true error=null syntax=1.0000 output=1.0000 similarity=1.0000 explanation=0.3667
```

---

## Validator Commands

```bash
# Full validation (includes Docker build)
python validate_submission.py

# Minimal mode — skip Docker and HF Space checks
python validate_submission.py --minimal

# Skip only Docker
python validate_submission.py --no-docker

# Include HF Space URL ping
python validate_submission.py --hf-url https://<username>-codedebug-env.hf.space

# Verbose output
python validate_submission.py --minimal --verbose
```

The validator runs 8 check groups:
1. Required environment variables
2. OpenEnv spec files and manifest sanity
3. Typed Pydantic models
4. Endpoint smoke tests (in-process, no server needed)
5. HF Space ping (optional)
6. Docker build success (optional)
7. inference.py runtime and structured logs
8. Task/grader count and score bounds

---

## Continuous Integration

[.github/workflows/ci.yml](.github/workflows/ci.yml) runs on every push, pull
request, and manual dispatch:

| Job | What it proves |
|-----|----------------|
| `validate` (Python 3.10 **and** 3.12) | The documented `pip install -e ".[dev]"` works on the minimum supported Python, `pytest` passes, and `validate_submission.py --minimal` passes |
| `docker` | The image builds, ships the tools its docs run (the validator executes *inside* the container), and actually serves traffic: the container is started and `scripts/smoke_test.py` checks `/api/v1/health`, `/reset`, `/state`, and `/step` over real HTTP before the run can pass |

The Docker job is the reason CI matters here: the image build cannot be verified
in every environment, so a broken image would otherwise reach a Hugging Face
Space unnoticed. No step is marked `continue-on-error`, so a failed build or an
unreachable health endpoint fails the run.

Run the same HTTP smoke test locally against any running server:

```bash
python app.py &                    # or: docker run -p 7860:7860 codedebug-env:latest
python scripts/smoke_test.py --base-url http://127.0.0.1:7860
# -> 4/4 checks passed, exit 0.  Exits 1 and says why if the server is down.
```

`scripts/smoke_test.py` is stdlib-only, so it needs no installs: CI runs it on a
runner where no project dependency exists yet, and the script is copied into the
image so it can also be run against a live server from inside a container:

```bash
docker run -d --name codedebug -p 7860:7860 codedebug-env:latest
docker exec codedebug python scripts/smoke_test.py --base-url http://localhost:7860
```

---

## Hugging Face Spaces Deployment

### Prerequisites
- Hugging Face account
- `huggingface_hub` installed: `pip install huggingface_hub`
- HF token with write access: `huggingface-cli login`

### Push to HF Spaces

```bash
# 1. Create Space at https://huggingface.co/spaces (type: Docker)

# 2. Add HF remote
git remote add hf https://huggingface.co/spaces/<username>/codedebug-env

# 3. Push
git push hf main

# 4. Monitor build logs in the Space UI

# 5. Verify deployment
curl https://<username>-codedebug-env.hf.space/api/v1/health
```

### Python push workflow
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path=".",
    repo_id="<username>/codedebug-env",
    repo_type="space",
    ignore_patterns=["*.pyc", "__pycache__", ".git", ".env"],
)
```

### HF Spaces `README.md` header (add to top of README for Spaces)
```yaml
---
title: CodeDebug OpenEnv
emoji: 🐛
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
```

---

## Pre-Submission Checklist

- [ ] `python validate_submission.py --minimal` → all checks pass
- [ ] `python inference.py --max-steps 1` → emits `[START]`, `[STEP]`, `[END]`, `[SUMMARY]`
- [ ] `python -m pytest tests/` → all regression tests pass
- [ ] `docker build -t codedebug-env .` → exits 0
- [ ] `docker run -p 7860:7860 codedebug-env` starts and `/health` returns 200
- [ ] `python scripts/smoke_test.py --base-url http://127.0.0.1:7860` → 4/4 checks passed
- [ ] GitHub Actions CI green on the pushed commit (`.github/workflows/ci.yml`)
- [ ] `openenv.yaml` present at repo root
- [ ] `manifest.json` present with all required keys
- [ ] At least **3 tasks** in `TASK_REGISTRY`
- [ ] At least **3 graders** in `GRADER_REGISTRY`
- [ ] All grader scores confirmed in `[0.0, 1.0]`
- [ ] HF Space URL responds to `/api/v1/health`
- [ ] `README.md` complete with quickstart instructions
- [ ] Course folder has 5 modules, each with README + notebook
- [ ] `git status` is clean (no uncommitted changes)
- [ ] Repository pushed to GitHub and Hugging Face

---

## Course Content

Five progressive modules in `course/`:

| Module | Topic |
|--------|-------|
| [01 — Intro to OpenEnv](course/module_01_intro/README.md) | Lifecycle, manifest, core concepts |
| [02 — Environment Design](course/module_02_environment/README.md) | Tasks, typed models, grader architecture |
| [03 — LLM Inference](course/module_03_inference/README.md) | OpenAI client, env vars, structured logs |
| [04 — Grader Engineering](course/module_04_graders/README.md) | Reward shaping, custom graders |
| [05 — Deployment](course/module_05_deployment/README.md) | Docker, HF Spaces, CI |

Each module has a `README.md` (concepts) and `notebook.ipynb` (hands-on exercises).

---

## License
MIT — see [LICENSE](LICENSE).
