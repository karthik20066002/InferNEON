#!/usr/bin/env bash
# test_io.sh — try a QEMU I/O config and report whether the guest progresses.
# Usage: test_io.sh "<drive-args>" <seconds>
set -uo pipefail
ROOT=/home/axiss/Documents/projects/InferNEON
LOG=/tmp/fw_test.log
ARGS=${1:-"format=raw"}
SECS=${2:-150}
cd "$ROOT" || exit 1

for p in $(pgrep -f "qemu-syste[m]"); do kill "$p" 2>/dev/null; done
sleep 2
rm -f "$LOG"

setsid qemu-system-aarch64 -M virt -cpu cortex-a72 -nographic -m 128 \
  -device loader,file=kernel.bin,addr=0x40100000,cpu-num=0 \
  -drive file=disk-olmoe-full.img,if=none,id=hd0,$ARGS \
  -device virtio-blk-device,drive=hd0 > "$LOG" 2>&1 < /dev/null &
QPID=$!
echo "qemu pid=$QPID args=[$ARGS]"

for i in $(seq 1 $((SECS/30))); do
  sleep 30
  if ! kill -0 "$QPID" 2>/dev/null; then echo "  t=$((i*30))s QEMU EXITED"; break; fi
  nf=$(grep -c "^fwd" "$LOG" 2>/dev/null || echo 0)
  nt=$(grep -c "^tok" "$LOG" 2>/dev/null || echo 0)
  rb=$(grep read_bytes /proc/$QPID/io 2>/dev/null | awk '{print $2}')
  rss=$(ps -o rss= -p "$QPID" 2>/dev/null | awk '{print int($1/1024)}')
  cpu=$(ps -o pcpu= -p "$QPID" 2>/dev/null | tr -d ' ')
  echo "  t=$((i*30))s fwd=$nf tok=$nt read=$((rb/1024/1024))MB rss=${rss}MB cpu=${cpu}%"
done
kill "$QPID" 2>/dev/null
echo "=== log tail:"; tail -n 3 "$LOG"
