"""Zero-shot (or fine-tuned) extraction eval on a Modal GPU.

Uses the EXACT app prompt (build_extract_prompt) + parser (parse_extraction) + metric,
so the number is directly comparable across base vs fine-tuned.

    modal run sft/modal_eval.py --model openbmb/MiniCPM5-1B --eval-file eval40_pd.jsonl
    modal run sft/modal_eval.py --model openbmb/MiniCPM5-1B --eval-file eval.jsonl   # full set
"""
import modal

app = modal.App("what-changed-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate")
    .add_local_dir("sft", "/root/sft")
    .add_local_dir("whatchanged", "/root/whatchanged")
    .add_local_dir("data/sft", "/root/data/sft")
)


@app.function(gpu="A10G", image=image, timeout=1800)
def evaluate(model_id: str, eval_file: str, n: int):
    import sys
    sys.path.insert(0, "/root")
    import json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from whatchanged.inference import build_extract_prompt, parse_extraction
    from sft.metric import dataset_f1, field_f1

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda")
    model.eval()

    preds, golds, rows = [], [], []
    with open(f"/root/data/sft/{eval_file}") as fh:
        for i, line in enumerate(fh):
            if n and i >= n:
                break
            row = json.loads(line)
            prompt = build_extract_prompt(row["note"])
            # RAW completion — matches training (sft/train.py) and the app's llama.cpp
            # create_completion path, so train / eval / serve are all consistent.
            enc = tok(prompt, return_tensors="pt")
            enc = {k: v.to("cuda") for k, v in enc.items()}
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=256, do_sample=False)
            raw = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = parse_extraction(raw)
            preds.append(pred)
            golds.append(row["label"])
            rows.append((row["note"], raw, pred, row["label"]))

    print(f"\n=== field-F1 ({model_id}) over {len(preds)} rows: "
          f"{dataset_f1(preds, golds):.3f} ===\n")
    for note, raw, pred, gold in rows[:12]:
        print(f"[F1 {field_f1(pred, gold)[2]:.2f}] {note[:80]}")
        print(f"   raw : {raw[:90]!r}")
        print(f"   pred: {pred}   gold: {gold}")


@app.local_entrypoint()
def run(model: str = "openbmb/MiniCPM5-1B", eval_file: str = "eval.jsonl", n: int = 0):
    evaluate.remote(model, eval_file, n)
