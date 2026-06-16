"""Back-extraction (round-trip) consistency audit of synthetic data.

For each (note, label), a strong extractor (MiniCPM4.1-8B via the free API, thinking off)
predicts the label FROM the note; we score field_f1 vs the gold label. High = the note
faithfully encodes the label (learnable); low = noisy/contradictory. Reports the
distribution and the worst rows (eyeball these to confirm real noise vs oracle error).

    export MINICPM_API_KEY=...
    python -m sft.audit --data data/sft/eval.jsonl --n 200
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
from whatchanged.inference import build_extract_prompt, parse_extraction
from sft.metric import field_f1, field_f1_tol


def make_extractor():
    from openai import OpenAI
    client = OpenAI(
        base_url=os.environ.get("MINICPM_BASE_URL", "http://35.203.155.71:8001/v1"),
        api_key=os.environ.get("MINICPM_API_KEY", "EMPTY"), timeout=60)
    model = os.environ.get("MINICPM_MODEL", "MiniCPM4.1-8B")

    def extract(note: str) -> dict:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": build_extract_prompt(note)}],
            max_tokens=160, temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return parse_extraction(r.choices[0].message.content or "")
    return extract


def main(data: str, n: int, worst: int):
    extract = make_extractor()
    rows = []
    for i, line in enumerate(open(data)):
        if n and i >= n:
            break
        row = json.loads(line)
        pred = extract(row["note"])
        exact = field_f1(pred, row["label"])[2]
        tol = field_f1_tol(pred, row["label"], tol=1)[2]
        rows.append((tol, exact, row["note"], pred, row["label"]))
    tols = [r[0] for r in rows]
    exacts = [r[1] for r in rows]
    print(f"\nConsistency over {len(rows)} rows (back-extraction vs gold):")
    print(f"   mean exact-match F1   = {statistics.mean(exacts):.3f}")
    print(f"   mean +/-1-tolerant F1 = {statistics.mean(tols):.3f}")
    for thr in (0.999, 0.75, 0.5):
        print(f"   rows with tol-F1 >= {thr:>5}: "
              f"{sum(1 for s in tols if s >= thr)}/{len(rows)}")
    print(f"\n--- {worst} worst by tolerant F1 (genuinely noisy / ambiguous) ---")
    for tol, exact, note, pred, gold in sorted(rows, key=lambda x: x[0])[:worst]:
        print(f"[tol {tol:.2f} | exact {exact:.2f}] {note[:90]}")
        print(f"   extracted: {pred}")
        print(f"   gold     : {gold}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft/eval.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--worst", type=int, default=12)
    a = ap.parse_args()
    main(a.data, a.n, a.worst)
