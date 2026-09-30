#!/usr/bin/env bash
# run_fw_matrix.sh — firmware generation for the extended verification matrix.
#
# For each prompt: rebuild the kernel with that prompt baked in (the firmware
# has no input path), then run it under the thermal watchdog and save the UART
# log. Three prompts x (16 prefill + 32 decode) forwards at ~100 s/forward
# under TCG is ~4 hours; the watchdog in run_qemu_tokens.sh pauses above 92 C
# and resumes below 80 C so this can run unattended.
#
# Each prompt's log is saved separately so a partial failure costs one prompt,
# not the whole matrix.
set -uo pipefail

ROOT=/home/axiss/Documents/projects/InferNEON
cd "$ROOT" || exit 1

# prompt_id:token_ids (16 tokens each; see tools/run_host_ref.py for the source)
PROMPTS=(
  "sum_calib_01:11808,4175,907,253,1563,2216,323,247,2718,16518,27,247,8050,14,33107,6956"
  "tamil_calib_01:47361,9053,108,42920,9053,114,42920,9053,112,23153,31165,115,42920,9053,113,42516"
  "math_calib_01:19932,422,253,3264,24671,5072,11061,591,10669,323,247,3387,14,12026,1755,14"
)

for entry in "${PROMPTS[@]}"; do
  pid="${entry%%:*}"
  ids="${entry#*:}"
  log="$ROOT/inferneon-eval/reports/fw_${pid}.log"

  if [ -s "$log" ] && grep -q "^generated:" "$log"; then
    echo "$(date +%H:%M) SKIP $pid (log already complete)" >> "$ROOT/inferneon-eval/reports/qemu_gate.log"
    continue
  fi

  echo "$(date +%H:%M) === $pid: rebuilding kernel" >> "$ROOT/inferneon-eval/reports/qemu_gate.log"
  if ! inferneon-eval/.venv/bin/python tools/gen_prompt.py "$ids" > /dev/null 2>&1; then
    echo "$(date +%H:%M) $pid: BUILD FAILED" >> "$ROOT/inferneon-eval/reports/qemu_gate.log"
    continue
  fi

  echo "$(date +%H:%M) === $pid: running firmware" >> "$ROOT/inferneon-eval/reports/qemu_gate.log"
  # run_qemu_tokens.sh writes qemu_tokens.log; move it aside per prompt.
  rm -f "$ROOT/inferneon-eval/reports/qemu_tokens.log"
  TIMEOUT=7200 bash scripts/run_qemu_tokens.sh
  mv "$ROOT/inferneon-eval/reports/qemu_tokens.log" "$log" 2>/dev/null
  ntok=$(grep -c "^tok:" "$log" 2>/dev/null || echo 0)
  echo "$(date +%H:%M) === $pid: done, $ntok tokens" >> "$ROOT/inferneon-eval/reports/qemu_gate.log"
done

echo "$(date +%H:%M) MATRIX COMPLETE" >> "$ROOT/inferneon-eval/reports/qemu_gate.log"
