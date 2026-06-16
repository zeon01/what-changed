"""Run the LoRA SFT on a Modal GPU (uses the hackathon $250 Modal credit).

One-time setup:
    pip install modal && modal token new

Generate data locally FIRST (no GPU; uses the free MiniCPM API):
    python -m sft.generate --n 1200 --out data/sft/train.jsonl --seed 0

Then train on Modal (data/sft is baked into the image):
    modal run sft/modal_train.py --base openbmb/MiniCPM5-1B          # PRIMARY
    modal run sft/modal_train.py --base nvidia/NVIDIA-Nemotron-3-Nano-4B  # fallback

Download the trained adapter from the Modal volume:
    modal volume get what-changed-sft-out <name> ./out/<name>
"""
import modal

app = modal.App("what-changed-sft")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Let unsloth pull its own compatible transformers / trl / peft / datasets / torch
    # (pinning them caused a peft<->unsloth conflict).
    .pip_install("unsloth==2026.5.1")
    # bake the training code + data into the image
    .add_local_dir("sft", "/root/sft")
    .add_local_dir("whatchanged", "/root/whatchanged")
    .add_local_dir("data/sft", "/root/data/sft")
)

vol = modal.Volume.from_name("what-changed-sft-out", create_if_missing=True)


@app.function(gpu="A10G", image=image, timeout=60 * 60, volumes={"/out": vol})
def train(base: str, epochs: int = 2, limit: int = 0):
    import sys
    sys.path.insert(0, "/root")
    import json
    import torch
    from unsloth import FastLanguageModel
    from sft.train import main as train_main
    from whatchanged.inference import build_extract_prompt, parse_extraction
    from sft.metric import dataset_f1, dataset_f1_tol

    name = base.split("/")[-1].lower()
    out = f"/out/{name}"
    model, tok = train_main(base, "/root/data/sft/train.jsonl", out, epochs, limit)
    vol.commit()
    print(f"Adapter committed to volume 'what-changed-sft-out' at {out}")

    # Eval the fine-tuned model immediately — RAW completion (matches training + app).
    FastLanguageModel.for_inference(model)
    preds, golds, samples = [], [], []
    with open("/root/data/sft/eval.jsonl") as fh:
        for line in fh:
            row = json.loads(line)
            enc = tok(build_extract_prompt(row["note"]), return_tensors="pt").to("cuda")
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=128, do_sample=False)
            raw = tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            preds.append(parse_extraction(raw))
            golds.append(row["label"])
            if len(samples) < 8:
                samples.append((row["note"][:70], raw[:90], preds[-1], row["label"]))
    print(f"=== FINE-TUNED {base}: exact-F1={dataset_f1(preds, golds):.3f} | "
          f"tol-F1={dataset_f1_tol(preds, golds):.3f} (n={len(preds)}) ===")
    for note, raw, pred, gold in samples:
        print(f"NOTE {note!r}\n  raw={raw!r}\n  pred={pred}  gold={gold}")


@app.local_entrypoint()
def run(base: str = "openbmb/MiniCPM5-1B", epochs: int = 2, limit: int = 0):
    train.remote(base, epochs, limit)
