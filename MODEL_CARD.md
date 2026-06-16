---
license: apache-2.0
base_model: openbmb/MiniCPM5-1B
library_name: gguf
language:
  - en
pipeline_tag: text-generation
tags:
  - parkinsons
  - healthcare
  - information-extraction
  - lora
  - gguf
  - llama-cpp
  - on-device
  - build-small-hackathon
---

# What Changed — Parkinson's caregiver-note extractor (fine-tuned MiniCPM5-1B)

A **1.08B** model fine-tuned to read a **family caregiver's daily free-text note** about a
parent with **Parkinson's disease** and return structured symptom fields as JSON. Built for the
Build Small Hackathon and designed to run **fully locally** via llama.cpp / GGUF — the whole
point is that health data logged at home never leaves the device.

**▶️ Try it live:** [the *What Changed* app on Spaces](https://huggingface.co/spaces/build-small-hackathon/what-changed) — daily logging (typed or spoken), trend charts, and a one-page doctor report, all running this model on-device.

> **Not medical advice / not a medical device.** It organizes a caregiver's own day-to-day
> observations into a structured form to share with a clinician. It does not diagnose, predict,
> or recommend treatment.

## What it does

**Input** — a short caregiver note:
> *"Mom moved with more struggle today, froze up twice, and her meds wore off before lunch."*

**Output** — compact JSON over a fixed schema:
```json
{"mobility": 2, "freezing_episodes": 2, "off_episodes": 1}
```
- Six daily ratings, **1–5 where 5 = best/healthiest**: `mobility`, `tremor`, `stiffness`,
  `mood`, `sleep`, `alertness`.
- Six event counts: `off_episodes`, `dyskinesia_spells`, `freezing_episodes`, `falls`,
  `missed_or_late_meds`, `hallucinations`.

Only fields the note actually mentions appear. The schema is aligned with the **Hauser PD Home
Diary** (ON/OFF + dyskinesia) and **MDS-UPDRS** motor/non-motor items. The model does exactly
one narrow job (note → JSON); all trend detection is deterministic Python downstream, so the
model never touches the reasoning that drives the report.

## Results — field-F1 (exact / within-±1)

| Eval set | exact | tol (±1) |
|---|---|---|
| In-distribution (200 held-out) | **0.97** | **0.99** |
| Out-of-distribution probe (25 deliberately-adversarial styles) | **0.83** | **0.91** |
| Base MiniCPM5-1B, zero-shot | ~0.00 | ~0.08 |

The base model can't do the task untrained; the capability is entirely from fine-tuning. (±1
calibration on a 1–5 scale is subjective and irrelevant to trend detection, so the **tol**
column is the one to trust.)

## Training

- **Method:** LoRA (r=16, α=16) via [Unsloth](https://github.com/unslothai/unsloth), bf16, 2
  epochs, ~80 s on a single A10G. Same RAW-completion prompt/parser at train, eval, and serve.
- **Data:** 1,296 synthetic `note → label` pairs, generated **label-first** (sample a
  ground-truth label, then have a teacher write a faithful caregiver note → labels correct by
  construction). Teacher: **Claude Sonnet 4.6**, under a strict third-person diary contract,
  plus a style-diverse robustness top-up (negation, terse/txt, slang, idioms).

## Running locally

Quantizations — field-F1 measured **through llama.cpp** on 80 held-out notes (the real serving path):

| file | size | exact | tol (±1) |
|---|---|---|---|
| `MiniCPM5-1B.Q8_0.gguf` ✅ recommended | 1.15 GB | 0.95 | 0.99 |
| `MiniCPM5-1B.Q4_K_M.gguf` smaller | 0.69 GB | 0.90 | 0.98 |

**Q8_0 is the pick** — it matches full precision (F16) at half the size. Q4_K_M trades ~5 points
of exact-match for ~0.46 GB if you need it on a very constrained device.

```bash
# llama.cpp CLI
llama-cli -m MiniCPM5-1B.Q8_0.gguf -n 96 -p "<build_extract_prompt(note)>"
```
```python
# llama-cpp-python — RAW completion, matches training/eval/serve
from llama_cpp import Llama
llm = Llama(model_path="MiniCPM5-1B.Q8_0.gguf", n_ctx=1024)
out = llm.create_completion(build_extract_prompt(note), max_tokens=96, temperature=0.0)
```
**Recommended:** constrain decoding with the project's **GBNF grammar** so output is always
valid schema JSON (correct keys, integer 1–5 / counts) — this also fixes rare out-of-schema
event slang at decode time.

## Prompt format

RAW completion (no chat template): the app's `build_extract_prompt(note)` is fed directly and
the model emits the JSON object. The exact template + parser live in the project repo.

## Limitations

- Trained on **synthetic** data and **never validated on real caregiver notes** (real notes are
  exactly the PII the project refuses to collect). The OOD probe is the best available proxy.
- Very abbreviated or heavily typo'd phrasings are the weakest spot; out-of-schema event slang
  is handled at serve time by the GBNF grammar, not the weights.
- English only.

---
*Built by **Saad Sharif Ahmed** ([@zeon01](https://huggingface.co/zeon01)) for the Build Small Hackathon. Code: [github.com/zeon01/what-changed](https://github.com/zeon01/what-changed)*
