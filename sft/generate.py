"""Generate note->label SFT data using the free MiniCPM API as the teacher.

Set the endpoint + key (defaults are the shared hackathon values):
    export MINICPM_BASE_URL="http://35.203.155.71:8001/v1"
    export MINICPM_API_KEY="sk-minicpm-...."          # from the OpenBMB slides
    export MINICPM_MODEL="MiniCPM4.1-8B"

Usage:
    python -m sft.generate --n 1200 --out data/sft/train.jsonl --seed 0
    python -m sft.generate --n 200  --out data/sft/eval.jsonl  --seed 99

MiniCPM4.1-8B is a reasoning model, so we pass enable_thinking=False and strip stray
<think> blocks + any "Note:" preamble. The prompt avoids concrete example phrases (the
model parrots them) and instead specifies tone + faithfulness rules.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import time
from sft.schema import sample_label, label_to_spec

_THINK = re.compile(r"(?s)<think>.*?</think>")
_PREFIX = re.compile(
    r"^\s*(?:here(?:'s| is)\b[^:\n]*:|a?\s*(?:short\s+)?note\b[^:\n]*:|"
    r"caregiver(?:'s)?\s+note\s*:)\s*",
    re.I,
)


def build_teacher_prompt(spec: str) -> str:
    return (
        "Jot a quick, casual note the way a tired family caregiver would about their "
        "elderly parent who has Parkinson's disease, over a day. Refer to them as 'Mom' or 'Dad'. One or two short "
        "sentences in everyday words (never clinical). Mention ONLY the facts below and "
        "nothing else, and reflect each fact faithfully and naturally. If an event is "
        "listed with a count, the note MUST say it happened that many times, in words "
        "(e.g. 'wore off a couple of times', 'fell once'). Do not use digits.\n"
        f"Facts: {spec}\n"
        "If the facts say 'nothing was recorded', write ONLY a brief remark that nothing "
        "specific stood out today (vary the wording; no example to copy; don't say they "
        "did well or badly). Otherwise, when facts ARE listed, write just those "
        "observations and NEVER tack on a 'nothing to note' phrase.\n"
        "Output the note text only — no preamble and no quotation marks.\n"
        "Note:"
    )


def make_teacher():
    """OpenAI-compatible client pointed at the free MiniCPM API. Returns a
    `complete(prompt) -> str` callable (cleans thinking + preamble, light retry).
    Imports `openai` lazily so the module is importable/testable without it."""
    from openai import OpenAI
    client = OpenAI(
        base_url=os.environ.get("MINICPM_BASE_URL", "http://35.203.155.71:8001/v1"),
        api_key=os.environ.get("MINICPM_API_KEY", "EMPTY"),
        timeout=60,
    )
    model = os.environ.get("MINICPM_MODEL", "MiniCPM4.1-8B")

    def complete(prompt: str, max_tokens: int = 120) -> str:
        last = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens, temperature=0.7,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                text = resp.choices[0].message.content or ""
                text = _THINK.sub("", text).strip()
                text = _PREFIX.sub("", text).strip().strip('"').strip()
                return text
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2 * (attempt + 1))
        raise last

    return complete


def generate(n: int, out: str, seed: int, teacher) -> None:
    """teacher is a `complete(prompt) -> str` callable (real API or a stub)."""
    rng = random.Random(seed)
    with open(out, "w") as fh:
        for _ in range(n):
            label = sample_label(rng)
            note = teacher(build_teacher_prompt(label_to_spec(label))).strip().strip('"')
            fh.write(json.dumps({"note": note, "label": label}) + "\n")
            fh.flush()  # survive an interrupted long run


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate(args.n, args.out, args.seed, make_teacher())
    print(f"Wrote {args.n} examples to {args.out}")
