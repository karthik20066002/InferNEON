# InferNEON

**InferNEON is a mechanism: fetch, cap, cancel, demand-beats-hint, max wait.**
**Policy stays above it.**
Qwen traces are the ruler. OLMoE-in-QEMU is proof the forward path exists.
The Pi is later.

That sentence defines the boundary, not a claim that every primitive is
implemented. The current task is measurement, not more mechanism.

## Current scope

- Commit the reference traces and the single-lane simulator in
  `inferneon-eval/` before extending anything.
- “Oracle” means demand-time perfect IDs only: `predict(j) = D_j`.
  Future-aware Belady eviction is a separate eviction comparison, not an oracle.
- Measure decode-only exposed wait on both Qwen trace families: LRU/no hint,
  B1 (previous token’s 8), and perfect IDs at matched `L ∈ {0, 1, 4, 40}`.
  Bandwidth `100e6` and `413e6` bytes/s and compute slack are **assumed**;
  start every prompt with an **empty cache**.
- Deliver `inferneon-eval/reports/exposed_wait_matched_L.json`, one
  exposed-wait plot, and a 10-line note. Capacity dominating or hints doing
  nothing is a result, not a reason to change the experiment.
- No cancel, in-flight cap, worst-case bound, new predictor, or `sim.engine`
  rewrite before that JSON is committed. After measurement, choose at most
  one primitive justified by it. Hardware waits for a laptop measurement;
  CM4 stays Debian-on-SD, with no SSD work.

## Matched-L measurement

- [Committed measurements](inferneon-eval/reports/exposed_wait_matched_L.json)
- [One exposed-wait plot](inferneon-eval/reports/exposed_wait_matched_L.svg)
- [10-line note](inferneon-eval/reports/exposed_wait_matched_L.md)

Under the labeled assumptions, B1 ties no-hint everywhere; capacity has a
large effect, and perfect IDs help at positive lookahead. That is headroom,
not a measured predictor win or a reason to add a primitive now.

Run measurement checks and reproduce sequentially (Python standard library only):

```sh
cd inferneon-eval
python3 -m unittest tests.test_queue tests.test_prefetch_deadline tests.test_exposed_wait_matched
python3 scripts/exposed_wait_matched_L.py
```

The full existing test suite also needs PyTorch; run it in the evaluation
virtual environment with `.venv/bin/python -m unittest discover -s tests`.

## Forward-path proof

Target (for now): QEMU `virt` machine, `-cpu cortex-a72`. No real hardware yet.
Storage is backed by a QEMU `virtio-blk` flat image. The OLMoE forward path
is a functional proof, not a storage-performance or hardware result.

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
- `src/attn.rs` — KV cache and attention.
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
