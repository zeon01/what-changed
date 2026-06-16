"""Evaluate a model on the held-out set (run on a GPU). Reuses the app's
build_extract_prompt + parse_extraction so eval matches production exactly.

    python -m sft.evaluate --model out/minicpm --eval data/sft/eval.jsonl
    python -m sft.evaluate --model openbmb/MiniCPM5-1B --eval data/sft/eval.jsonl  # base
"""
from __future__ import annotations
import argparse
import json
from whatchanged.inference import build_extract_prompt, parse_extraction
from sft.metric import dataset_f1


def hf_complete(model, tokenizer, prompt: str) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main(model_path: str, eval_path: str):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path, max_seq_length=1024, load_in_4bit=True)
    FastLanguageModel.for_inference(model)

    preds, golds = [], []
    with open(eval_path) as fh:
        for line in fh:
            row = json.loads(line)
            raw = hf_complete(model, tokenizer, build_extract_prompt(row["note"]))
            preds.append(parse_extraction(raw))
            golds.append(row["label"])
    print(f"{model_path}: dataset_f1 = {dataset_f1(preds, golds):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", default="data/sft/eval.jsonl")
    a = ap.parse_args()
    main(a.model, a.eval)
