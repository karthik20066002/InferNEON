# InferNEON — bare-metal AArch64 MoE inference

Target (for now): QEMU `virt` machine, `-cpu cortex-a72`. No real hardware yet.
Storage is backed by a QEMU `virtio-blk` flat image (`disk.img`); the
flat-blob-at-fixed-offsets design is identical to the real-hardware plan so
the later eMMC/PCIe swap is a backend swap, not a redesign.

Toolchain: Rust `no_std`, `aarch64-unknown-none`, `target-cpu=cortex-a72`.
Build with `just` (see `justfile`). Output is a **flat binary**
(`kernel.bin` via `llvm-objcopy -O binary`); `kernel.elf` is kept for GDB.

Legacy C prototype lives in `legacy/` — untouched, owned separately.
Host-side eval tooling lives in `inferneon-eval/` — separate project, not
part of the firmware build.

## Layout

- `src/main.rs` — entry (`kernel_main`), panic handler.
- `src/boot.rs` — `_start`: FP enable, vectors, per-core stacks (MPIDR), BSS zero, park secondaries.
- `src/{token,layer,gate,dispatch,cache,exec,combine,head,pipeline}.rs` — MoE pipeline steps 1–7.
- `src/attn.rs` — sole KV/attention stub (NOT DESIGNED YET).
- `src/storage/{mod,virtio,emmc,pcie}.rs` — `BlockStorage` trait + backends.
- `src/{uart,addr,sync,config}.rs` — platform support.
- `docs/STORAGE_LAYOUT.md` — expert flat-blob spec (layer table + arithmetic).
- `tools/fetch_olmoe.sh`, `tools/pack_olmoe.py` — host-side model fetch/pack (verify uniform-expert invariant).

## Quick loop

```sh
just build            # neon=1 default: NEON kernels on aarch64
just build 0          # neon=0: scalar reference backend (--no-default-features)
just build 1 4096     # neon on, eMMC-width experiment (stride recomputed at export)
just test             # host unit tests (scalar path; NEON is aarch64-only)
just mkdisk           # 64M stub disk.img
just qemu             # boots kernel.bin, virtio-blk attached
```

Boot log reports `exec backend: neon|scalar` + `exec selftest: PASS`
(dot/GEMV exact constants, SiLU sanity, NEON-vs-scalar cross-check on
aarch64). Boot enables FP/SIMD via `CPACR_EL1` (`src/boot.rs`) — required
before any `f32`/NEON instruction.

## Pipeline (see src/pipeline.rs)

1. Token queue (FIFO) → 2. Layer counter (bounded index) → 3. Gating top-k →
4. Dispatch unordered set + barrier → 5. Expert cache resolve (MFU/LRU) →
6. Join on matmul outputs → 7. Last-layer? output head/sample/drain : next layer.
