"""Zero-shot baseline of a base HF model on CPU (no training, no GPU).

Uses the EXACT app prompt (build_extract_prompt) + parser (parse_extraction), so the
number is directly comparable to the fine-tuned model. Meant for a SUBSET (--n) on CPU.

    python -m sft.baseline_hf --model openbmb/MiniCPM5-1B --n 40
"""
from __future__ import annotations
import argparse
import json
from whatchanged.inference import build_extract_prompt, parse_extraction
from sft.metric import dataset_f1, field_f1


def _load(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()
    return tok, model


def _complete(tok, model, prompt: str, max_new_tokens: int = 64) -> str:
    import torch
    try:
        enc = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            return_tensors="pt", return_dict=True)
    except Exception:
        enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main(model_id: str, eval_path: str, n: int, show: int):
    tok, model = _load(model_id)
    preds, golds, rows = [], [], []
    for i, line in enumerate(open(eval_path)):
        if n and i >= n:
            break
        row = json.loads(line)
        raw = _complete(tok, model, build_extract_prompt(row["note"]))
        pred = parse_extraction(raw)
        preds.append(pred)
        golds.append(row["label"])
        rows.append((row["note"], raw, pred, row["label"]))
    print(f"\nBASELINE (zero-shot, {model_id}) field-F1 over {len(preds)} rows: "
          f"{dataset_f1(preds, golds):.3f}\n")
    for note, raw, pred, gold in rows[:show]:
        print(f"[F1 {field_f1(pred, gold)[2]:.2f}] {note[:80]}")
        print(f"   raw : {raw[:90]!r}")
        print(f"   pred: {pred}   gold: {gold}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", default="data/sft/eval.jsonl")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()
    main(a.model, a.eval, a.n, a.show)
