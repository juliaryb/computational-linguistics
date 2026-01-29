#!/usr/bin/env python3
"""
Run evaluation for a single task JSON against two models and prompting strategies.
Outputs:
- results/raw.jsonl : one record per model+strategy run
- results/scores.csv: scoring template (created if it doesn't exist)

Usage example:
  python scripts/run_eval.py --task prompts/ethical_reasoning.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --- Ollama client import ---
try:
    import ollama  # pip install ollama
except ImportError as e:
    raise SystemExit(
        "Missing dependency: ollama\n"
        "Install with: pip install ollama\n"
        "Then re-run this script."
    ) from e


# ----------------------------
# Prompt builders
# ----------------------------

def build_zero_shot(task: Dict[str, Any]) -> str:
    desc = task["description"].strip()
    eval_input = task["eval_example"]["input"].strip()
    return f"{desc}\n\nScenario:\n{eval_input}\n"


def build_few_shot(task: Dict[str, Any]) -> str:
    desc = task["description"].strip()
    dev_examples = task.get("dev_examples", [])
    if len(dev_examples) < 2:
        raise ValueError("Few-shot requires at least 2 dev_examples in the task JSON.")

    parts = [desc, "", "Here are some examples:"]
    for i, ex in enumerate(dev_examples[:3], start=1):
        ex_in = ex["input"].strip()
        ex_out = ex["output"].strip()
        parts.append(f"\nExample {i}:\nInput:\n{ex_in}\nOutput:\n{ex_out}")

    eval_input = task["eval_example"]["input"].strip()
    parts.append(f"\nNow solve this:\nInput:\n{eval_input}\nOutput:")
    return "\n".join(parts).strip() + "\n"


def build_cot(task: Dict[str, Any]) -> str:
    # CoT is only for the standard (non-reasoning) model in this assignment.
    desc = task["description"].strip()
    eval_input = task["eval_example"]["input"].strip()
    cot_instruction = (
        "Let's think step by step before answering, then provide a concise final answer.\n"
        "Format:\n"
        "Reasoning: <your step-by-step reasoning>\n"
        "Final: <your final answer>\n"
    )
    return f"{desc}\n\n{cot_instruction}\nScenario:\n{eval_input}\n"


# ----------------------------
# Ollama call
# ----------------------------

def query_ollama(model: str, prompt: str, options: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (response_text, raw_metadata).
    """
    # ollama.generate returns a dict-like object with keys like:
    # 'response', 'model', 'created_at', 'done', 'eval_count', 'eval_duration', etc.
    res = ollama.generate(model=model, prompt=prompt, options=options)
    text = res.get("response", "")
    return text, dict(res)


# ----------------------------
# IO helpers
# ----------------------------

def read_task_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        task = json.load(f)
    # Minimal validation
    required = ["task_name", "description", "dev_examples", "eval_example"]
    for k in required:
        if k not in task:
            raise ValueError(f"Task JSON missing required key: {k}")
    if "input" not in task["eval_example"]:
        raise ValueError("Task JSON eval_example must contain 'input'.")
    return task



def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Path to a single task JSON, e.g. prompts/ethical_reasoning.json")
    parser.add_argument("--out", default="results/raw.jsonl", help="JSONL output path (append). Default: results/raw.jsonl")
    
    # Models (defaults match your plan)
    parser.add_argument("--small_model", default="qwen2.5:1.5b", help="Standard small model for zero/few/cot.")
    parser.add_argument("--reasoning_model", default="deepseek-r1:7b", help="Reasoning model for zero/few (no cot).")

    # Generation options (keep conservative for consistency)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    task_path = Path(args.task)
    out_path = Path(args.out)

    task = read_task_json(task_path)

    # Build prompts per strategy
    strategy_builders = {
        "zero_shot": build_zero_shot,
        "few_shot": build_few_shot,
        "cot": build_cot,
    }

    # Assignment logic: CoT only for the standard (small) model, not for reasoning model
    runs: List[Tuple[str, str]] = [
        (args.small_model, "zero_shot"),
        (args.small_model, "few_shot"),
        (args.small_model, "cot"),
        (args.reasoning_model, "zero_shot"),
        (args.reasoning_model, "few_shot"),
    ]

    options = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_predict": args.max_tokens,
        "seed": args.seed,
    }

    for model, strategy in runs:
        prompt = strategy_builders[strategy](task)

        run_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat(timespec="seconds")

        t0 = time.time()
        response_text, raw_meta = query_ollama(model=model, prompt=prompt, options=options)
        t1 = time.time()

        record = {
            "run_id": run_id,
            "timestamp": timestamp,
            "task_file": str(task_path),
            "task_name": task.get("task_name", ""),
            "model": model,
            "strategy": strategy,
            "options": options,
            "prompt": prompt,
            "response": response_text,
            "runtime_seconds": round(t1 - t0, 3),
            "ollama_meta": {
                # keep only a subset to avoid huge logs; adjust if you want more
                "model": raw_meta.get("model"),
                "created_at": raw_meta.get("created_at"),
                "done": raw_meta.get("done"),
                "total_duration": raw_meta.get("total_duration"),
                "load_duration": raw_meta.get("load_duration"),
                "prompt_eval_count": raw_meta.get("prompt_eval_count"),
                "prompt_eval_duration": raw_meta.get("prompt_eval_duration"),
                "eval_count": raw_meta.get("eval_count"),
                "eval_duration": raw_meta.get("eval_duration"),
            },
        }

        append_jsonl(out_path, record)
        print(f"[OK] {task['task_name']} | {model} | {strategy} | run_id={run_id} | {record['runtime_seconds']}s")

    print(f"\nWrote outputs to: {out_path}")


if __name__ == "__main__":
    main()
