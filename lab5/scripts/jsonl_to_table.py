import json
import csv

input_path = "results/raw.jsonl"
output_path = "results/outputs_for_scoring.csv"

fields = [
    "run_id",
    "task_name",
    "model",
    "strategy",
    "response",
    "runtime_seconds"
]

with open(input_path, "r", encoding="utf-8") as f, \
     open(output_path, "w", newline="", encoding="utf-8") as out:

    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()

    for line in f:
        r = json.loads(line)
        writer.writerow({
            "run_id": r["run_id"],
            "task_name": r["task_name"],
            "model": r["model"],
            "strategy": r["strategy"],
            "response": r["response"].strip(),
            "runtime_seconds": r["runtime_seconds"]
        })