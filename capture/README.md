# Capture — Phase B prep (torch-dependent run, code ready before torch lands)
# Run only after: torch+accelerate+gptqmodel installed, model weights approved.
# Entry: `capture/run_capture.py --prompt_file data/prompts/code.jsonl --out data/traces/<rev>/ --max_new_tokens 512 --seed 0`
# Records per decode token: 40x top-8 + weights + 256 logits fp16, h_in + h_postmix (calib + 2 heldout/domain), router weights.
# Validation: `scripts/validate_traces.py` recomputes top-8 from h_postmix+router+norm, requires >=99.9% match.

print("capture stub: waiting on torch + weights approval")
