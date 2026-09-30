#!/usr/bin/env bash
# diagnose_block.sh — launch the firmware, let it block, then collect
# everything about what QEMU is waiting on.
set -uo pipefail
ROOT=/home/axiss/Documents/projects/InferNEON
LOG=/tmp/fw_diag.log
cd "$ROOT" || exit 1

for p in $(pgrep -f "qemu-syste[m]"); do kill "$p" 2>/dev/null; done
sleep 2
rm -f "$LOG"

setsid qemu-system-aarch64 -M virt -cpu cortex-a72 -nographic -m 128 \
  -device loader,file=kernel.bin,addr=0x40100000,cpu-num=0 \
  -drive file=disk-olmoe-full.img,if=none,id=hd0,format=raw \
  -device virtio-blk-device,drive=hd0 > "$LOG" 2>&1 < /dev/null &
QPID=$!
echo "qemu pid=$QPID, waiting 90s for it to block..."
sleep 90

echo "=== guest log tail:"
tail -n 3 "$LOG"
echo "=== read_bytes: $(grep read_bytes /proc/$QPID/io 2>/dev/null)"
echo "=== cpu: $(ps -o pcpu= -p $QPID 2>/dev/null)"
echo "=== main thread wchan: $(cat /proc/$QPID/wchan 2>/dev/null)"
echo "=== threads (comm + wchan):"
for t in /proc/$QPID/task/*; do
  printf "  %-16s %s\n" "$(cat $t/comm 2>/dev/null)" "$(cat $t/wchan 2>/dev/null)"
done 2>/dev/null | head -n 8
echo "=== strace 3s (syscalls in flight):"
timeout 3 strace -p "$QPID" -f 2>&1 | head -n 15
echo "=== host disk during block:"
iostat -d 1 2 2>/dev/null | tail -n 4
echo "=== can the host read the image right now?"
dd if=disk-olmoe-full.img of=/dev/null bs=1M count=200 iflag=direct 2>&1 | tail -n 1

kill "$QPID" 2>/dev/null
