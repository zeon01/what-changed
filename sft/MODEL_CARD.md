---
license: apache-2.0
base_model: openbmb/MiniCPM5-1B
tags: [what-changed, minicpm, openbmb, structured-extraction, gguf, llama-cpp, build-small-hackathon]
---

# What Changed — note→JSON extractor (MiniCPM5-1B, fine-tuned)

Fine-tuned for ONE task: turning a caregiver's free-text note about an elderly person's
day into structured JSON (six domains rated 1–5 + event counts). Built for the **What
Changed** app (Build Small Hackathon, Backyard AI track). Runs fully locally via llama.cpp.

- **Base:** `openbmb/MiniCPM5-1B` (~1.08B params, llama architecture).
- **Method:** LoRA SFT on ~1.2k synthetic `note → JSON` pairs (labels sampled first, then
  paraphrased into natural notes by the MiniCPM4.1-8B teacher) — no real or scraped data.
- **Prompt:** identical to the app's inference prompt (train/serve consistency).
- **Held-out field-F1:** <fill from the bake-off table>.
- **Not for diagnosis.** Output is descriptive structured data only; the app's trend
  detection is deterministic, not model-driven.

## Usage (llama.cpp / llama-cpp-python)

```python
from llama_cpp import Llama
llm = Llama(model_path="what-changed-1b-q4_k_m.gguf", n_ctx=4096)
# Feed the app's build_extract_prompt(note); parse the JSON object from the output.
```
