#!/usr/bin/env bash
# smoke_test.sh — verify the instrumented firmware produces parseable lg/tok
# output before committing hours to the full matrix.
set -uo pipefail
ROOT=/home/axiss/Documents/projects/InferNEON
LOG=/tmp/fw_smoke.log
cd "$ROOT" || exit 1

for p in $(pgrep -f "qemu-syste[m]"); do kill "$p" 2>/dev/null; done
sleep 2

rm -f "$LOG"
# cache=none: the 13.8 GiB image cannot fit in 14 GiB of host RAM, so the
# default writeback cache thrashes (kswapd + swap) and the guest stalls in
# poll_schedule_timeout instead of computing. Direct I/O sidesteps that.
setsid qemu-system-aarch64 -M virt -cpu cortex-a72 -nographic -m 128 \
  -device loader,file=kernel.bin,addr=0x40100000,cpu-num=0 \
  -drive file=disk-olmoe-full.img,if=none,id=hd0,format=raw,cache=none \
  -device virtio-blk-device,drive=hd0 > "$LOG" 2>&1 < /dev/null &
QPID=$!
echo "qemu pid=$QPID"

# Watch for the first lg/tok group (first forward + head pass), up to 20 min.
for i in $(seq 1 40); do
  sleep 30
  if grep -q "^tok:" "$LOG" 2>/dev/null; then
    echo "=== first token after ~$((i*30))s:"
    grep -E "^(fwd|head shard|lg|tok)" "$LOG" | head -n 16
    kill "$QPID" 2>/dev/null
    exit 0
  fi
  cpu=$(ps -o pcpu= -p "$QPID" 2>/dev/null | tr -d ' ')
  rb=$(grep read_bytes /proc/$QPID/io 2>/dev/null | awk '{print $2}')
  echo "  t=$((i*30))s cpu=${cpu}% read_bytes=${rb} lines=$(wc -l < "$LOG")"
done
echo "=== TIMEOUT: no token in 20 min"
tail -n 5 "$LOG"
kill "$QPID" 2>/dev/null
exit 1
