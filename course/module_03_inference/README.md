# Module 03 — LLM Inference

## Connecting an LLM Agent to OpenEnv

```
for each task:
    env.reset(task_id)            ← [START] line emitted here
    for each step:
        agent.act(observation) -> Action
        score all graders
        env.step(action)          ← [STEP]  line emitted here
        if done: break
                                  ← [END]   line emitted here
```

## OpenAI-Compatible Client

```python
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("HF_TOKEN"),
)
response = client.chat.completions.create(
    model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2,
    max_tokens=512,
)
```

## Environment Variables

| Variable       | Purpose                  | Example |
|----------------|--------------------------|---------|
| API_BASE_URL   | API endpoint base URL    | https://api.openai.com/v1 |
| MODEL_NAME     | Model identifier         | gpt-4o-mini |
| HF_TOKEN       | HF token used as API key | hf_xxx... |

```bash
export API_BASE_URL=https://api-inference.huggingface.co/models/meta-llama/Llama-3.3-70B-Instruct/v1
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export HF_TOKEN=hf_your_token_here
python inference.py --tasks task_syntax_001 task_logic_001
```

## Structured Output Format

`inference.py` prints one line per event to **stdout** (flushed immediately).
Every line begins with the tag as the first characters — no JSON wrapping.

```
[START] task=task_syntax_001 env=codedebug-env model=gpt-4o-mini
[STEP] step=1 action=def multiply(a, b):\n    return a * b\n reward=0.91 done=true error=null
[END] task=task_syntax_001 success=true solved=true steps=1 score=0.91 rewards=0.91
[START] task=task_logic_001 env=codedebug-env model=gpt-4o-mini
[STEP] step=1 action=def fibonacci(n):\n    ...\n reward=0.91 done=true error=null
[END] task=task_logic_001 success=true solved=true steps=1 score=0.91 rewards=0.91
[SUMMARY] total=2 solved=2 avg_score=0.9059 elapsed=0.1s within_limit=true
```

### Why plain-text (not JSON)?
The OpenEnv validator scans stdout for lines that **start with** the tag string
(`[START]`, `[STEP]`, `[END]`).  Wrapping the tag inside a JSON value hides it
from the scanner — the tags must be the literal first characters of each line.

### Verbose mode — per-grader detail
```bash
python inference.py --verbose
# [STEP] step=1 action=... reward=0.91 done=true error=null syntax=1.0000 output=1.0000 similarity=1.0000 explanation=0.3667
```
The four trailing fields (`syntax`, `output`, `similarity`, `explanation`) are
the individual grader scores that make up the composite reward.

> `score` is the **best** composite score reached during the episode, and
`solved`/`success` reflect whether the fix passed every test case — matching
the environment's own solve criterion.

### Grep / awk on the output
```bash
# Show only END lines
python inference.py | grep "^\[END\]"

# Extract scores
python inference.py | awk '/^\[END\]/ {print $3, $4}'

# Count solved tasks
python inference.py | grep "^\[END\]" | grep "solved=True" | wc -l
```

## Running inference.py

```bash
# Default: first 3 tasks, no LLM key needed (uses fallback agent)
python inference.py

# Specific tasks
python inference.py --tasks task_syntax_001 task_logic_001 task_runtime_001

# Override steps + save summary JSON
python inference.py --max-steps 2 --output results.json

# Verbose: see per-grader breakdown on every [STEP] line
python inference.py --verbose
```

## Prompt Engineering Tips
1. Ask the model to respond in **strict JSON only** — no markdown fences
2. Keep `temperature` low (0.1–0.3) for deterministic fixes
3. Always include the error message in the prompt — it is the biggest hint
4. Test with `--verbose` to inspect per-grader scores and tune your prompt

-> Next: [Module 04 — Grader Engineering](../module_04_graders/README.md)
