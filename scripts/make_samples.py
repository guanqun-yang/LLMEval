import argparse
import json
import math
import os
import random
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lm_eval_path = os.path.join(repo_root, "lm-evaluation-harness")
if lm_eval_path not in sys.path:
    sys.path.insert(0, lm_eval_path)

from lm_eval.tasks import TaskManager
from lm_eval.evaluator_utils import get_task_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--pct", type=float, default=0.05)
    parser.add_argument("--min_cap", type=int, default=10)
    parser.add_argument("--max_cap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", type=str, default="*")
    parser.add_argument("--include_path", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)

    task_manager = TaskManager(include_path=args.include_path)
    patterns = [t.strip() for t in args.tasks.split(",") if t.strip()]
    matched = task_manager.match_tasks(patterns)

    samples = {}
    for name in matched:
        try:
            task_dict = task_manager.load_task_or_group(name)
            outputs = get_task_list(task_dict)
        except Exception:
            continue

        for out in outputs:
            task_obj = out.task
            if not task_obj:
                continue
            try:
                n = len(task_obj.eval_docs)
            except Exception:
                continue
            if n <= 0:
                continue
            k = int(math.ceil(n * args.pct))
            k = max(k, args.min_cap)
            k = min(k, args.max_cap, n)
            indices = sorted(random.sample(range(n), k))
            samples[out.task_name] = indices

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(samples, f)


if __name__ == "__main__":
    main()