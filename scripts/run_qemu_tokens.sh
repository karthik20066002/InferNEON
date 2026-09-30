#!/usr/bin/env bash
# run_qemu_tokens.sh — supervised token-generation run.
#
# Two things the bare `timeout qemu` invocation could not do:
#
#  1. Visibility. A QEMU-TCG forward pass over 16 layers x 8 experts is
#     minutes, and the whole proof rides on whether 7 of them fit the budget.
#     The firmware prints per-forward and per-head-shard heartbeats, so pace
#     is measured, not guessed.
#
#  2. Thermal safety. Sustained TCG load holds the SoC at 84C+ and this box
#     has hard-shutdown-tripped twice before ("critical temperature reached").
#     Rather than gamble an unattended multi-hour run, the watchdog SIGSTOPs
#     QEMU above --pause-at and SIGCONTs it back below --resume-below. Pausing
#     is lossless: the guest resumes with its RAM, KV cache and block-device
#     state exactly as they were.
#
# A thermal abort is recorded in the log; a timeout is recorded as such. Both
# are distinguishable from a real crash, which matters for how the result is
# later described.
set -uo pipefail

ROOT=/home/axiss/Documents/projects/InferNEON
LOG=$ROOT/inferneon-eval/reports/qemu_tokens.log
GATE=$ROOT/inferneon-eval/reports/qemu_gate.log
TIMEOUT=${TIMEOUT:-14400}
PAUSE_AT=${PAUSE_AT:-92000}      # milli-degrees C: 92 C
RESUME_BELOW=${RESUME_BELOW:-80000}  # 80 C

cd "$ROOT" || exit 1

if pgrep -f "qemu-system-aarch64.*disk-olmoe-full" > /dev/null; then
  echo "$(date +%H:%M) REFUSING: another qemu holds the image lock" >> "$GATE"
  exit 1
fi

echo "$(date +%H:%M) FIRING qemu (instrumented; timeout ${TIMEOUT}s, pause>${PAUSE_AT})" >> "$GATE"

timeout "$TIMEOUT" qemu-system-aarch64 -M virt -cpu cortex-a72 -nographic -m 128 \
  -device loader,file=kernel.bin,addr=0x40100000,cpu-num=0 \
  -drive file=disk-olmoe-full.img,if=none,id=hd0,format=raw \
  -device virtio-blk-device,drive=hd0 > "$LOG" 2>&1 &
QPID=$!

# Watchdog: poll thermal_zone0, pause/resume the QEMU process group.
paused=0
while kill -0 "$QPID" 2>/dev/null; do
  temp=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 0)
  if [ "$paused" -eq 0 ] && [ "$temp" -gt "$PAUSE_AT" ]; then
    kill -STOP "$QPID" 2>/dev/null
    paused=1
    echo "$(date +%H:%M) THERMAL PAUSE at ${temp}mC (resumes below ${RESUME_BELOW}mC)" >> "$GATE"
  elif [ "$paused" -eq 1 ] && [ "$temp" -lt "$RESUME_BELOW" ]; then
    kill -CONT "$QPID" 2>/dev/null
    paused=0
    echo "$(date +%H:%M) THERMAL RESUME at ${temp}mC" >> "$GATE"
  fi
  sleep 20
done

wait "$QPID"
rc=$?
echo "$(date +%H:%M) qemu exit rc=$rc" >> "$GATE"
