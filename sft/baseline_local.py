"""Zero-shot baseline of a local GGUF model on the eval set (CPU, no training).

Uses the EXACT app prompt (build_extract_prompt) + parser (parse_extraction), so the
number is directly comparable to the fine-tuned model later.

Usage:
    python -m sft.baseline_local --model models/minicpm5-1b-q4.gguf --n 50
"""
from __future__ import annotations
import argparse
import json
from whatchanged.inference import LlamaCppBackend, build_extract_prompt, parse_extraction
from sft.metric import dataset_f1, field_f1


def main(model_path: str, eval_path: str, n: int, show: int):
    backend = LlamaCppBackend(model_path, n_ctx=2048)
    preds, golds, rows = [], [], []
    for i, line in enumerate(open(eval_path)):
        if n and i >= n:
            break
        row = json.loads(line)
        raw = backend.complete(build_extract_prompt(row["note"]), max_tokens=128)
        pred = parse_extraction(raw)
        preds.append(pred)
        golds.append(row["label"])
        rows.append((row["note"], pred, row["label"]))
    print(f"\nBASELINE (zero-shot) field-F1 over {len(preds)} rows: "
          f"{dataset_f1(preds, golds):.3f}\n")
    for note, pred, gold in rows[:show]:
        print(f"[F1 {field_f1(pred, gold)[2]:.2f}] {note[:90]}")
        print(f"   pred: {pred}")
        print(f"   gold: {gold}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", default="data/sft/eval.jsonl")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()
    main(a.model, a.eval, a.n, a.show)
