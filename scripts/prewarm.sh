#!/usr/bin/env bash
# prewarm.sh — warm the page cache in the FIRMWARE's read order.
#
# The image is 12.9 GiB and the box has 14 GiB RAM with ~6 GiB held by
# desktop apps, so the whole image cannot be cached. A sequential pre-warm
# caches the first ~5.8 GiB (early experts) and evicts the non-expert tensors
# at the end — exactly the wrong end, because the firmware reads those FIRST
# (embed, routers, norms, attn, head) and on every single forward.
#
# So: warm the non-expert region (read every forward) and the first few
# layers of experts (read first), which is where the blocking was observed.
set -uo pipefail
IMG=/home/axiss/Documents/projects/InferNEON/disk-olmoe-full.img
BS=1M

# Non-expert tensors: 12.88 GiB .. end (embed, routers, norms, attn, head, rope).
echo "warming non-expert region (12.88 GiB .. end)..."
dd if="$IMG" of=/dev/null bs=$BS skip=12288 2>&1 | tail -n 1

# First 4 GiB of experts (layers 0..4, read first by the firmware).
echo "warming first 4 GiB of experts..."
dd if="$IMG" of=/dev/null bs=$BS count=4096 2>&1 | tail -n 1

echo "cache now: $(grep -E '^Cached' /proc/meminfo | awk '{printf "%.1f GiB", $2/1048576}')"
