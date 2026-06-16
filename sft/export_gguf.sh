#!/usr/bin/env bash
# Merge the LoRA adapter into the base, convert to GGUF, and quantize to Q4_K_M.
# Usage: sft/export_gguf.sh <adapter_dir> <base_model> <out_name>
# Example: sft/export_gguf.sh out/minicpm openbmb/MiniCPM5-1B what-changed-1b
#
# MiniCPM5-1B is llama-architecture, so llama.cpp's convert_hf_to_gguf.py handles it
# natively. (An official BASE gguf exists at openbmb/MiniCPM5-1B-GGUF, but we export OUR
# fine-tuned merge.)
set -euo pipefail
ADAPTER="$1"; BASE="$2"; NAME="$3"

# 1) merge LoRA into base -> merged/
python - "$ADAPTER" "$BASE" <<'PY'
import sys
from unsloth import FastLanguageModel
adapter, base = sys.argv[1], sys.argv[2]
model, tok = FastLanguageModel.from_pretrained(model_name=adapter, max_seq_length=1024)
model.save_pretrained_merged("merged", tok, save_method="merged_16bit")
print("merged ->", "merged/")
PY

# 2) convert to GGUF + quantize (requires a local llama.cpp checkout)
: "${LLAMA_CPP:?set LLAMA_CPP to your llama.cpp checkout dir}"
python "$LLAMA_CPP/convert_hf_to_gguf.py" merged --outfile "models/${NAME}-f16.gguf"
"$LLAMA_CPP/llama-quantize" "models/${NAME}-f16.gguf" "models/${NAME}-q4_k_m.gguf" Q4_K_M
echo "GGUF -> models/${NAME}-q4_k_m.gguf"
