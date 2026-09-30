#!/usr/bin/env bash
# verify_new_kernel.sh — confirm the rebuilt MAX_TOKENS=128 kernel boots and
# reaches the forward loop, and report whether it progresses.
set -uo pipefail
ROOT=/home/axiss/Documents/projects/InferNEON
LOG=/tmp/fw_verify.log
cd "$ROOT" || exit 1

# Kill any stray QEMU by exact binary name (bracket avoids self-match).
for p in $(pgrep -x qemu-system-aarch64); do kill "$p" 2>/dev/null; done
sleep 2
rm -f "$LOG"

echo "kernel: $(md5sum kernel.bin | cut -c1-12)  image: $(md5sum disk-olmoe-full.img | cut -c1-12)"
echo "source MAX_TOKENS: $(grep -o 'MAX_TOKENS: usize = [0-9]*' src/config.rs)"

setsid qemu-system-aarch64 -M virt -cpu cortex-a72 -nographic -m 128 \
  -device loader,file=kernel.bin,addr=0x40100000,cpu-num=0 \
  -drive file=disk-olmoe-full.img,if=none,id=hd0,format=raw \
  -device virtio-blk-device,drive=hd0 > "$LOG" 2>&1 < /dev/null &
QPID=$!
echo "qemu pid=$QPID"

for i in $(seq 1 10); do
  sleep 30
  if ! kill -0 "$QPID" 2>/dev/null; then echo "  t=$((i*30))s QEMU EXITED"; break; fi
  nf=$(grep -c "^fwd" "$LOG" 2>/dev/null || echo 0)
  nt=$(grep -c "^tok" "$LOG" 2>/dev/null || echo 0)
  rb=$(grep read_bytes /proc/$QPID/io 2>/dev/null | awk '{print $2}')
  cpu=$(ps -o pcpu= -p "$QPID" 2>/dev/null | tr -d ' ')
  echo "  t=$((i*30))s fwd=$nf tok=$nt read=$((rb/1024/1024))MB cpu=${cpu}%"
  if [ "$nt" -gt 0 ]; then echo "  *** TOKEN PRODUCED ***"; break; fi
done
kill "$QPID" 2>/dev/null
echo "=== log:"
cat "$LOG"
