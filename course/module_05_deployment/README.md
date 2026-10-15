# Module 05 — Deployment

## Local Development

```bash
pip install -e ".[dev]"
uvicorn openenv_env.server:create_app --factory --port 8000 --reload
# or
python app.py
```

## Docker

```bash
# Build
docker build -t codedebug-env:latest .

# Run
docker run -p 7860:7860 codedebug-env:latest

# With env vars
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=hf_xxx \
  codedebug-env:latest
```

## Hugging Face Spaces

1. Create a new Space (type: Docker) at https://huggingface.co/spaces
2. Push your repo:
```bash
git remote add hf https://huggingface.co/spaces/<username>/codedebug-env
git push hf main
```
3. HF Spaces auto-builds your Dockerfile and exposes port 7860.
4. Confirm the space is live:
```bash
curl https://<username>-codedebug-env.hf.space/api/v1/health
```

## GitHub Actions CI

The real workflow is checked in at `.github/workflows/ci.yml` — run it on every
push, pull request, and manual dispatch. It has two jobs:

- **validate** — matrix over Python 3.10 and 3.12: `pip install -e ".[dev]"`,
  `python -m pytest tests/ -q`, `python validate_submission.py --minimal`.
- **docker** — builds the image, runs `validate_submission.py --minimal` *inside*
  it (proving the image ships the tools its docs invoke), starts the container,
  smoke-tests `/api/v1/health`, `/reset`, `/state`, and `/step` over real HTTP
  with `scripts/smoke_test.py`, then waits for the container `HEALTHCHECK` to
  report `healthy`.

No step uses `continue-on-error`, so a failed build or a server that never
answers fails the run.

Reproduce the HTTP smoke test against any running server:

```bash
python app.py &
python scripts/smoke_test.py --base-url http://127.0.0.1:7860
```

`scripts/smoke_test.py` uses only the standard library, so it needs no installs —
which is why CI can run it before any project dependency exists.

## Pre-Submission Checklist
- [ ] python validate_submission.py --minimal passes all checks
- [ ] python inference.py emits [START], [STEP], [END] tags
- [ ] Docker build succeeds locally (CI also builds it on every push)
- [ ] `python scripts/smoke_test.py --base-url <url>` reports 4/4 checks passed
- [ ] `.github/workflows/ci.yml` green on the pushed commit
- [ ] HF Space URL responds to /api/v1/health
- [ ] manifest.json has all required keys
- [ ] README is complete and accurate

## Troubleshooting

| Problem               | Fix |
|-----------------------|-----|
| Port already in use   | Change PORT env var |
| HF Space build fails  | Check requirements.txt for conflicts |
| Inference crashes     | Ensure API_BASE_URL and MODEL_NAME are set |
| Low rewards           | Review grader weights in graders.py |
