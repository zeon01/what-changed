---
title: What Changed
emoji: 📋
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
python_version: "3.11"
pinned: false
license: apache-2.0
short_description: Local Parkinson's caregiver diary to a doctor report
models:
  - zeon01/what-changed-1b
tags:
  - track:backyard
  - sponsor:openbmb
  - achievement:offgrid
  - achievement:welltuned
  - achievement:offbrand
  - achievement:llama
  - achievement:fieldnotes
---

# What Changed 📋

A private, **local** tracker that turns a caregiver's daily notes about a parent with
**Parkinson's disease** into a one-page **doctor report** — running entirely on a fine-tuned
**1B** model ([`zeon01/what-changed-1b`](https://huggingface.co/zeon01/what-changed-1b)) via
llama.cpp. Nothing leaves the device.

> **Not medical advice / not a medical device.** It organizes a caregiver's own observations
> to share with a clinician; it does not diagnose, predict, or recommend treatment.

## Build Small Hackathon — submission (track: Backyard AI)
- **Demo video:** [Watch the demo (Loom)](https://www.loom.com/share/59cb34d06f1b4a109ca03e2b3c836cd6)
- **📣 Social post:** [LinkedIn](https://www.linkedin.com/posts/saad-sharif-ahmed_what-changed-a-hugging-face-space-by-build-small-hackathon-activity-7472433691974238209-otI_)
- **👤 Built by:** Saad Sharif Ahmed — HF username: [zeon01](https://huggingface.co/zeon01)

## How it works
1. **Log** a day in plain words — *"froze in the doorway, pill wore off before lunch"* — and the
   fine-tuned 1B extracts structured ratings + event counts. Decoding is **grammar-constrained**,
   so the output is always valid schema JSON.
2. **Trends** are computed by **deterministic Python** (rolling averages, decline streaks, event
   rates) — the model never touches the math, so it can't hallucinate a trend.
3. **Report** narrates the already-computed findings in plain language to bring to the doctor.

A 1.08B model is enough because it only does two narrow jobs (note→JSON, and phrasing findings);
the reasoning is all deterministic. That's what makes it run locally on plain hardware.

*This demo is pre-seeded with a realistic 60-day decline arc so the Trends and Report tabs are
populated. Built for the Build Small Hackathon.*
