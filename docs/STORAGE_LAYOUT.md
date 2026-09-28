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
