#!/usr/bin/env bash
# Full decode capture: all 35 prompts x N greedy tokens, crash-resumable.
# Calibration prompts first (unblocks Helen's B0/B2), then heldout.
# Each prompt appends to its own data/traces/<id>.jsonl with per-token
# fsync; resume replays and verifies token ids. Completed prompts print
# COMPLETE and are skipped, so a crash costs at most the in-flight token.
# Logs: reports/capture_<id>.log (live-tail these),
#       reports/capture_master.log (START/DONE/COMPLETE per prompt).
set -u
cd "$(dirname "$0")/.."
TOKENS="${TOKENS:-128}"
export TOKENS
MASTER="reports/capture_master.log"
mkdir -p data/traces reports
export PROMPTS_LIST
PROMPTS_LIST="$(mktemp)"
.venv/bin/python -c "
import json, glob, os
def key(item):
    f, pid, split = item
    # calibration first (unblocks policy building), then heldout; alpha within
    return (0 if split == 'calibration' else 1, f, pid)
rows = []
with open(os.environ['PROMPTS_LIST'], 'w') as out:
    for f in sorted(glob.glob('data/prompts/*.jsonl')):
        for line in open(f):
            line = line.strip()
            if line:
                d = json.loads(line)
                rows.append((f, d['prompt_id'], d['split']))
    for f, pid, split in sorted(rows, key=key):
        out.write(f + ' ' + pid + '\n')
print(str(len(rows)) + ' prompts x ' + os.environ['TOKENS'] + ' tokens', flush=True)
"
echo "CAPTURE START $(date -Iseconds) tokens=$TOKENS" >> "$MASTER"
while read -r file pid; do
  log="reports/capture_${pid}.log"
  echo "START $pid $(date -Iseconds)" >> "$MASTER"
  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python scripts/capture_decode.py \
    --prompts-file "$file" --prompt-id "$pid" \
    --decode-tokens "$TOKENS" > "$log" 2>&1
  echo "DONE $pid rc=$? $(date -Iseconds)" >> "$MASTER"
done < "$PROMPTS_LIST"
rm -f "$PROMPTS_LIST"
echo "CAPTURE ALL DONE $(date -Iseconds)" >> "$MASTER"
