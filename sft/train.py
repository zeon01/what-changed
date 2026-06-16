"""LoRA SFT for note->JSON. Run on a GPU (locally or via sft/modal_train.py on Modal):
    python -m sft.train --base openbmb/MiniCPM5-1B --out out/minicpm        # PRIMARY
    python -m sft.train --base nvidia/NVIDIA-Nemotron-3-Nano-4B --out out/nemotron  # fallback

Training input is the SAME prompt the app uses at inference (build_extract_prompt); the
target is the gold JSON, so the model learns exactly the deployed task.
"""
from __future__ import annotations
import argparse
import json
from whatchanged.inference import build_extract_prompt


def load_pairs(path: str):
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            yield row["note"], row["label"]


def to_text(note: str, label: dict, eos: str) -> str:
    target = json.dumps(label, separators=(",", ":"))
    return f"{build_extract_prompt(note)} {target}{eos}"


def main(base: str, train_path: str, out: str, epochs: int = 2, limit: int = 0):
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    # bf16 (not 4-bit): a 1B fits easily on an A10G, so skip QLoRA's quantization error.
    # Serving precision (bf16/f16/Q8/Q4 GGUF) is a separate, post-hoc choice — TBD.
    # max_seq_length=512: our longest training example is ~240 tokens (2x+ headroom).
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base, max_seq_length=512, load_in_4bit=False)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    eos = tokenizer.eos_token or ""
    pairs = list(load_pairs(train_path))
    if limit:
        pairs = pairs[:limit]  # dry run / subset
    texts = [to_text(n, l, eos) for n, l in pairs]
    ds = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        args=SFTConfig(output_dir=out, num_train_epochs=epochs,
                       per_device_train_batch_size=8, learning_rate=2e-4,
                       logging_steps=20, save_strategy="epoch", dataset_text_field="text"))
    trainer.train()
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"Saved adapter to {out}")
    return model, tokenizer


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--train", default="data/sft/train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    main(a.base, a.train, a.out, a.epochs, a.limit)
