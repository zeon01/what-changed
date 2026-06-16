# What Changed 📋

> Turn a caregiver's one-sentence daily note into a doctor-ready view of how a parent with Parkinson's is changing — running entirely on a local, fine-tuned 1B model.

**▶️ Live demo:** https://huggingface.co/spaces/build-small-hackathon/what-changed
**🧠 Model:** https://huggingface.co/zeon01/what-changed-1b
Built for the [Build Small Hackathon](https://huggingface.co/build-small-hackathon) (Gradio + Hugging Face) — **Backyard AI** track.

> **Not medical advice / not a medical device.** It organizes a caregiver's own observations to share with a clinician; it does not diagnose, predict, or recommend treatment.

## What it does

A family caregiver describes a day in plain words — *"Dad froze in the doorway twice and barely slept"* — by **typing or speaking** it. A fine-tuned **1B model** extracts structured ratings (mobility, tremor, stiffness, mood, sleep, alertness; 1–5 where 5 = best) and event counts (OFF episodes, dyskinesia spells, freezing, falls, missed/late meds, hallucinations). Deterministic Python computes the trends, and one click produces a one-page **Doctor Report**.

The schema is aligned with the **Hauser PD Home Diary** (ON/OFF + dyskinesia) and **MDS-UPDRS** motor/non-motor items.

## Why it works at 1B

The model does exactly two narrow jobs — note→JSON extraction, and phrasing already-computed findings. **All trend detection is deterministic Python**, so the model never touches the math and can't hallucinate a trend. That's what lets a 1.08B model run the whole thing locally.

- **Local-first** — fine-tuned **MiniCPM5-1B** via llama.cpp / GGUF; no cloud APIs, nothing leaves the device.
- **Voice in, on-device** — speech-to-text via faster-whisper, also local.
- **Grammar-constrained decoding** — extraction is always valid schema JSON.

See [`FIELD_NOTES.md`](FIELD_NOTES.md) for the full build story — the data-quality saga, the CPU-deploy fixes, and the chart-design decisions.

## Layout

- `app.py` — Gradio app (Log / Trends / Doctor Report).
- `whatchanged/` — schema, SQLite store, deterministic trend engine, charts, inference (llama.cpp), STT, report, UI.
- `sft/` — synthetic-data pipeline + training / eval / export scripts (Modal).
- `scripts/seed_demo.py` — a realistic 60-day demo arc.
- `tests/` — unit tests (`pytest`).

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py            # downloads the published GGUF from the Hub on first run
```
Pre-seed demo data so Trends/Report are populated: `WC_SEED_DEMO=1 python app.py`.

## Tech

Fine-tuned **MiniCPM5-1B** (OpenBMB) · llama.cpp · faster-whisper · Gradio · Hugging Face · trained on Modal.

— Built by **Saad Sharif Ahmed** ([@zeon01](https://huggingface.co/zeon01)).
