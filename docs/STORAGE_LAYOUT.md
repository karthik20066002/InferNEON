# EXPERT STORAGE LAYOUT — flat blob, table + arithmetic hybrid

No filesystem from the kernel entry point onward. One flat blob on the block
device (today: QEMU `virtio-blk` `disk.img`; later: real eMMC/NVMe with the
same offsets). All multi-byte fields little-endian.

## 0. Boot-chain disclaimer (CM4, eventual target)

The BCM2711 boot ROM + SPI EEPROM firmware requires a small FAT32 boot
partition to find the first-stage kernel image. That is fixed vendor firmware
behavior, unrelated to and outside the control of this OS. Everything from
the kernel entry point onward — including this layout — is flat binary, no
filesystem, as designed.

## 1. Layer table at a fixed root

- LBA 0 (byte offset 0) of the blob holds the LAYER TABLE header + entries.
- Header (16 bytes, packed):
  - `magic: u32 = 0x0000AE86` — sanity check (wrong model / corrupt flash).
    16-bit `0xAE86` signature zero-extended to 32 bits for alignment.
  - `num_layers: u32` (OLMoE-1B-7B: 16).
  - `expert_stride_bytes: u32` — one expert's exported size ROUNDED UP at
    export time to `STORAGE_BLOCK_SIZE`. Checked: `stride % block == 0`.
  - `reserved: u32 = 0`.
- Followed by `num_layers` × `u64` little-endian entries:
  `layer_base[layer_id]` = byte offset of layer `layer_id`'s first expert.
- The table contents are GENERATED at export/build time by
  `tools/pack_olmoe.py`, not hand-hardcoded per layer. One hardcoded thing:
  the table's own address (LBA 0). Inter-layer padding is absorbed by the
  per-layer base entries without touching lookup code.

## 2. Intra-layer arithmetic (no per-expert table)

```text
expert_offset(layer_id, expert_index)
  = layer_table[layer_id] + expert_index * expert_stride_bytes
```

`expert_stride_bytes` makes every expert start block-aligned as a side
effect — no per-expert padding logic at runtime. Lookup is O(1) pure
arithmetic after one table indirection.

## 3. Uniform-expert-size invariant (OLMoE-1B-7B)

`hidden_size=2048, intermediate_size=1024`, identical across all 64 experts
× all 16 layers → single global `expert_bytes`, no per-expert override in
the model schema. `pack_olmoe.py` MUST re-verify this from the model's
actual safetensors headers (3072 tensors walked) and abort on any variance.
If the target model ever changes, re-verify before reusing pure arithmetic —
do not assume it silently transfers.

Per-expert param count (SwiGLU: gate+up+down):
`3 * 2048 * 1024 = 6_291_456` params; bytes = params × dtype width
(bf16 → 12,582,912 B). Stride = `round_up(expert_bytes, STORAGE_BLOCK_SIZE)`.

## 5. Qwen3.6-35B-A3B profile (`INFERNEON_MODEL=qwen`)

Numbers from the real upstream config (`text_config`, `qwen3_5_moe_text`):
hidden 2048, 40 layers, 256 routed experts of moe_intermediate 512, top-8
routed + one always-on shared expert (512-wide SwiGLU), silu. Same LBA-0
header format (magic `0xAE86`); per-expert stride 6291456 (3*2048*512*2,
512-aligned).

Upstream layout deltas, all absorbed at export (`tools/pack_qwen.py`) so the
kernel and its `expert_offset` math are unchanged:
- Routed gate+up ship FUSED as `experts.gate_up_proj` [256,1024,2048]; the
  packer splits the fused rows back into gate/up blobs. Kernel sees gate/up/down only.
- One shared expert per layer (gate/up [512,2048], down [2048,512]) at
  `shared_offset(layer) = layer_base + 256*expert_stride`, after the routed
  region. Dispatch joins it implicitly alongside top-8 (TODO in dispatch.rs).
- Router `mlp.gate.weight` [256,2048] + `shared_expert_gate` travel with the
  gating bundle, outside the expert blob region.
- Tensor prefix is `model.language_model.` (multimodal wrapper).

SCOPE: upstream BF16 layout. The eval team's GPTQ-INT4 artifact (group 128)
packs qweights differently — INT4 repack + dequant kernels are an explicit
TODO, not covered by this profile.

## 6. Block width

- QEMU `virtio-blk-device` default logical/physical sector: **512 B**.
  `STORAGE_BLOCK_SIZE` defaults to 512 and is build-configurable
  (`just build 1 4096` → `build.rs` → `config::STORAGE_BLOCK_SIZE`).
- The packer takes the same width and rounds the stride; the kernel asserts
  `stride % STORAGE_BLOCK_SIZE == 0` at boot. Mismatched image/kernel is a
  hard boot error, not a silent misalignment.
- Real CM4 eMMC erase-group/block geometry and PCIe Gen2 x1 ceilings are the
  eventual target, not now — same layout, different width/backend.

## 7. Non-expert region + directory (`tools/pack_olmoe_full.py`, disk-olmoe-full.img)

- The expert blob is byte-for-byte the v1 image (plain copy, so old images
  still boot to the storage probe). Everything else the forward pass needs
  lives AFTER the expert region, described by a directory in the first 512 B
  of that appended region.
- **The directory is not at a fixed LBA.** The v1 packer put `base0` at
  `round_up(16 + 16*8, 512)` = byte 512, so LBA 1 is expert (0,0)'s first
  512 bytes. Writing the directory there looked fine and silently corrupted
  one expert — the boot probe printed `expert00 head: 0x0000ae87`, i.e. the
  directory magic, where bf16 weights should be. The kernel therefore
  *derives* the directory address from the LBA-0 table it already trusts:
  `dir_lba = base0/512 + layers*experts*stride/512`
  (`config::dir_lba_after_experts`). Nothing about the appended region is
  hardcoded anywhere.
- The 512 B directory block is reserved before any tensor is emitted, and the
  packer asserts every entry starts at or after `v1_size + 512` — otherwise
  embed (the first tensor) would begin exactly where the directory goes.
- Directory contents: `magic2 u32 = 0x0000AE87`, `count u32`, then per entry
  `tag u32, offset u64, len u64, dtype u32, pad 4x` = **28 bytes** (the pad is
  part of the stride; reading 24-byte entries desynchronises every field after
  the first and only tag 1 resolves).
- Tags: 1 embed [50304,2048] bf16; 2 routers 16×[64,2048] bf16; 3 norms
  33×[2048] as stored (bf16); 4 attn per-layer q,k,v,o [4,2048,2048] bf16;
  5 head [50304,2048] bf16; 6 rope_cos [16,64] fp32; 7 rope_sin [16,64]
  fp32; 8 qk_norms 32×[2048] bf16 (rows q0,k0,q1,k1…).
- RoPE tables are host-exact fp32 (theta 10000, head_dim 128, Llama-style
  half rotation — validated against transformers); firmware trig was
  rejected (Taylor error ~0.2 at π flips routes). Tag 8 exists because
  `q_norm`/`k_norm` are *learned* RMSNorms applied to the full 2048-vector
  before the head reshape; omitting them diverges the routes.
- MAXPOS = 16 tokens (prompt + short decode + margin); KV sized to match.
- Peak static footprint ≈ 86 MiB (expert 12.6 + attn 33.6 + head 25.7 + KV 8
  + pinned 9), inside the 127 MiB link region — checked at build time.
