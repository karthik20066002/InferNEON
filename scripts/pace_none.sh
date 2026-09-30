#!/usr/bin/env bash
# pace_none.sh — measure firmware forward pace with cache=none (direct I/O),
# sampling read_bytes to get the actual guest disk read rate.
set -uo pipefail
ROOT=/home/axiss/Documents/projects/InferNEON
LOG=/tmp/fw_none.log
MINUTES=${1:-5}
cd "$ROOT" || exit 1

for p in $(pgrep -f "qemu-syste[m]"); do kill "$p" 2>/dev/null; done
sleep 2
rm -f "$LOG"

setsid qemu-system-aarch64 -M virt -cpu cortex-a72 -nographic -m 128 \
  -device loader,file=kernel.bin,addr=0x40100000,cpu-num=0 \
  -drive file=disk-olmoe-full.img,if=none,id=hd0,format=raw,cache=none \
  -device virtio-blk-device,drive=hd0 > "$LOG" 2>&1 < /dev/null &
QPID=$!
echo "qemu pid=$QPID (cache=none), watching ${MINUTES} min"

prev=0
for i in $(seq 1 "$MINUTES"); do
  sleep 60
  nf=$(grep -c "^fwd" "$LOG" 2>/dev/null || echo 0)
  nt=$(grep -c "^tok" "$LOG" 2>/dev/null || echo 0)
  cpu=$(ps -o pcpu= -p "$QPID" 2>/dev/null | tr -d ' ')
  rb=$(grep read_bytes /proc/$QPID/io 2>/dev/null | awk '{print $2}')
  rate=$(( (rb - prev) / 60 / 1024 / 1024 ))
  echo "  t=${i}min fwd=$nf tok=$nt cpu=${cpu}% read=$((rb/1024/1024))MB (+${rate}MB/s)"
  prev=$rb
done
kill "$QPID" 2>/dev/null
