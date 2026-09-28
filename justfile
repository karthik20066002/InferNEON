# InferNEON justfile — QEMU virt build/boot/test loop.
# Usage: just build / just build 0 / just build 1 4096 / just qemu / just mkdisk

default:
    @just --list

# Build the kernel (release, max-opts).
# neon=1 (default) enables the NEON kernels; neon=0 builds the scalar reference.
# block = storage sector width in bytes (QEMU virtio-blk default 512).
# model = olmoe (default) or qwen (Qwen3.6-35B-A3B profile).
build neon="1" block="512" model="olmoe":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{neon}}" = "1" ]; then FLAGS=""; else FLAGS="--no-default-features"; fi
    STORAGE_BLOCK_SIZE={{block}} INFERNEON_MODEL={{model}} cargo build --release $FLAGS
    llvm-objcopy -O binary target/aarch64-unknown-none/release/inferneon kernel.bin
    cp target/aarch64-unknown-none/release/inferneon kernel.elf
    ls -la kernel.bin kernel.elf
    echo "STORAGE_BLOCK_SIZE={{block}} neon={{neon}} model={{model}} (QEMU virtio-blk default block is 512)"

# Host-side unit tests (NEON path is aarch64-only; host exercises scalar).
test:
    cargo test --target x86_64-unknown-linux-gnu

# Host-side checks (target build).
check:
    cargo check --target aarch64-unknown-none

clean:
    cargo clean
    rm -f kernel.bin kernel.elf disk.img

# Make a flat backing image for the virtio-blk device (contents = flat blob, see docs/STORAGE_LAYOUT.md).
mkdisk size="64M":
    dd if=/dev/zero of=disk.img bs=1M count={{trim_end_match(size, "M")}} status=progress
    @ls -la disk.img

# Boot a flat binary under QEMU virt (default: stub disk.img).
# img=disk-olmoe.img boots the full packed expert image instead.
qemu img="disk.img": build
    qemu-system-aarch64 -M virt -cpu cortex-a72 -nographic \
      -device loader,file=kernel.bin,addr=0x40100000,cpu-num=0 \
      -drive file={{img}},if=none,id=hd0,format=raw \
      -device virtio-blk-device,drive=hd0

# Same but with the ELF (symbol-friendly, GDB).
qemu-elf: build
    qemu-system-aarch64 -M virt -cpu cortex-a72 -nographic \
      -kernel kernel.elf \
      -drive file=disk.img,if=none,id=hd0,format=raw \
      -device virtio-blk-device,drive=hd0

# GDB stub: terminal 1 `just qemu-gdb`, terminal 2 `aarch64-none-elf-gdb kernel.elf -ex 'target remote :1234'`.
qemu-gdb: build
    qemu-system-aarch64 -M virt -cpu cortex-a72 -nographic -S -s \
      -device loader,file=kernel.bin,addr=0x40100000,cpu-num=0 \
      -drive file=disk.img,if=none,id=hd0,format=raw \
      -device virtio-blk-device,drive=hd0

# Stamp the verified LBA-0 layer table onto the stub disk image.
stamp-table model="olmoe":
    python3 ./tools/stamp_table.py disk.img --model {{model}}

fetch-olmoe:
    ./tools/fetch_olmoe.sh

pack-olmoe:
    python3 ./tools/pack_olmoe.py --help
