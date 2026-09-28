#!/usr/bin/env bash
# Fetch OLMoE-1B-7B snapshot (host-side only; kernel never parses safetensors).
set -euo pipefail
MODEL="${MODEL:-allenai/OLMoE-1B-7B-0924}"
OUT="${OUT:-models/olmoe-1b-7b}"
echo "Fetching ${MODEL} -> ${OUT}"
echo "Requires: hf CLI (pipx install 'huggingface_hub[cli]')"
mkdir -p "${OUT}"
hf download "${MODEL}" --repo-type model --local-dir "${OUT}" --exclude "*.pt" || {
  echo "TIP: set MODEL=allenai/OLMoE-1B-7B-0125 for the newer release." >&2
  exit 1
}
echo "Done. Verify uniform-expert invariant, then run pack_olmoe.py."
