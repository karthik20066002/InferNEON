#!/usr/bin/env bash
# 35 prompts x {fast, forced-reference} deterministic route dumps.
# Resumable: completed outputs are skipped, so a crash costs one run, not all.
# Logs: reports/drift_35/<pid>_{fla,ref}.log (live-tail these),
#       reports/drift_35_master.log (START/DONE per run).
# Finish with: .venv/bin/python scripts/summarize_drift.py
set -u
cd "$(dirname "$0")/.."
# OUTDIR override keeps reruns as separate evidence (default: original dir).
OUTDIR="${OUTDIR:-reports/drift_35}"
MASTER="${OUTDIR}_master.log"
mkdir -p "$OUTDIR"
export PROMPTS_LIST
PROMPTS_LIST="$(mktemp)"
.venv/bin/python -c "
import json, glob, os
n = 0
def key(f):
    # mechanism-first: tamil_hindi pairs complete earliest for the gap probe
    return (0 if 'tamil' in f else 1, f)
with open(os.environ['PROMPTS_LIST'], 'w') as out:
    for f in sorted(glob.glob('data/prompts/*.jsonl'), key=key):
        for line in open(f):
            line = line.strip()
            if line:
                out.write(f + ' ' + json.loads(line)['prompt_id'] + '\n')
                n += 1
print(str(n) + ' prompts', flush=True)
"
echo "DRIFT35 START $(date -Iseconds) outdir=$OUTDIR" >> "$MASTER"
while read -r file pid; do
  for k in fla ref; do
    out="$OUTDIR/${pid}_${k}.json"
    log="$OUTDIR/${pid}_${k}.log"
    if [ -s "$out" ]; then
      echo "SKIP $pid $k (exists)" >> "$MASTER"
      continue
    fi
    if [ "$k" = "ref" ]; then extra="--force-reference-kernels"; else extra=""; fi
    echo "START $pid $k $(date -Iseconds)" >> "$MASTER"
    # shellcheck disable=SC2086
    CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python scripts/dump_routes.py \
      --tokens 1 --deterministic $extra \
      --prompts-file "$file" --prompt-id "$pid" --out "$out" > "$log" 2>&1
    # shellcheck disable=SC2181
    echo "DONE $pid $k rc=$? $(date -Iseconds)" >> "$MASTER"
  done
done < "$PROMPTS_LIST"
rm -f "$PROMPTS_LIST"
echo "DRIFT35 ALL DONE $(date -Iseconds)" >> "$MASTER"
