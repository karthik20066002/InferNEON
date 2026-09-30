#!/usr/bin/env bash
# measure_pace.sh — run the firmware for N minutes and report forwards/min,
# so the full matrix can be sized on evidence instead of a guess.
set -uo pipefail
ROOT=/home/axiss/Documents/projects/InferNEON
LOG=/tmp/fw_pace.log
MINUTES=${1:-6}
cd "$ROOT" || exit 1

for p in $(pgrep -f "qemu-syste[m]"); do kill "$p" 2>/dev/null; done
sleep 2
rm -f "$LOG"

setsid qemu-system-aarch64 -M virt -cpu cortex-a72 -nographic -m 128 \
  -device loader,file=kernel.bin,addr=0x40100000,cpu-num=0 \
  -drive file=disk-olmoe-full.img,if=none,id=hd0,format=raw \
  -device virtio-blk-device,drive=hd0 > "$LOG" 2>&1 < /dev/null &
QPID=$!
echo "qemu pid=$QPID, watching ${MINUTES} min (default cache)"

for i in $(seq 1 "$MINUTES"); do
  sleep 60
  nf=$(grep -c "^fwd" "$LOG" 2>/dev/null || echo 0)
  nt=$(grep -c "^tok" "$LOG" 2>/dev/null || echo 0)
  cpu=$(ps -o pcpu= -p "$QPID" 2>/dev/null | tr -d ' ')
  rb=$(grep read_bytes /proc/$QPID/io 2>/dev/null | awk '{print $2}')
  echo "  t=${i}min fwd=$nf tok=$nt cpu=${cpu}% read_bytes=$((rb/1024/1024))MB"
done

kill "$QPID" 2>/dev/null
echo "=== last lines:"
tail -n 4 "$LOG"
