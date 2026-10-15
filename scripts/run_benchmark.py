import os, sys
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from openenv_env.evaluator import run_benchmark
from openenv_env.tasks import TASK_REGISTRY
from openenv_env.models import Action

def solver(obs, tid):
    t=TASK_REGISTRY[tid]
    return Action(fixed_code=t.fixed_code, explanation="", confidence=1.0)

res=run_benchmark(solver)
import json; print(json.dumps(res["summary"], indent=2))
