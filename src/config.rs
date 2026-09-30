//! config.rs — single source of tunable constants (nothing hardcoded in logic).
//!
//! Two model profiles, selected at build time (`INFERNEON_MODEL=olmoe|qwen`,
//! default olmoe; `just build 1 512 qwen`). Profile deltas come from the real
//! upstream configs (OLMoE-1B-7B; Qwen3.6-35B-A3B text_config, qwen3_5_moe).
//! STORAGE_BLOCK_SIZE is build-configurable (build.rs, default 512).

const fn str_eq(a: &str, b: &str) -> bool {
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    if ab.len() != bb.len() {
        return false;
    }
    let mut i = 0;
    while i < ab.len() {
        if ab[i] != bb[i] {
            return false;
        }
        i += 1;
    }
    true
}

const MODEL_ENV: Option<&str> = option_env!("INFERNEON_MODEL");

const fn is_qwen() -> bool {
    match MODEL_ENV {
        Some(s) => str_eq(s, "qwen"),
        None => false,
    }
}

/// True for the Qwen3.6-35B-A3B profile, false for OLMoE-1B-7B.
pub const MODEL_IS_QWEN: bool = is_qwen();

/// Human-readable profile tag for the boot log.
pub const MODEL_NAME: &str = if MODEL_IS_QWEN {
    "qwen3.6-35B-A3B"
} else {
    "olmoe-1B-7B"
};

/// MoE layers: OLMoE 16, Qwen 40.
pub const NUM_LAYERS: u32 = if MODEL_IS_QWEN { 40 } else { 16 };
/// Routed experts per layer: OLMoE 64, Qwen 256.
pub const NUM_EXPERTS_PER_LAYER: usize = if MODEL_IS_QWEN { 256 } else { 64 };
/// Active routed experts per token: top-8 in BOTH profiles.
pub const TOP_K: usize = 8;
/// Upper bound for any supported top-k (static sizing, no alloc).
pub const MAX_K: usize = 16;
/// Cores brought up at boot for v1 (design for N; see boot.rs / sync.rs).
pub const NUM_CORES_BRINGUP: usize = 1;
/// Hidden width: 2048 in BOTH profiles.
pub const HIDDEN_SIZE: usize = 2048;
/// Per-expert intermediate width: OLMoE 1024, Qwen moe_intermediate 512.
pub const INTERMEDIATE_SIZE: usize = if MODEL_IS_QWEN { 512 } else { 1024 };
/// Qwen routes one extra always-on shared expert (same SwiGLU shape as a
/// routed expert). OLMoE has none. Dispatch must join it implicitly (TODO).
pub const HAS_SHARED_EXPERT: bool = MODEL_IS_QWEN;
/// Routed-expert bf16 bytes in the upstream layout: 3*hidden*inter*2.
/// OLMoE 12582912; Qwen 6291456 (fused gate_up is SPLIT at export — the
/// kernel only ever sees gate/up/down, see docs/STORAGE_LAYOUT.md).
pub const EXPERT_BYTES: usize = if MODEL_IS_QWEN { 6291456 } else { 12582912 };

/// Storage medium block/sector width in bytes (QEMU virtio-blk default 512).
pub const STORAGE_BLOCK_SIZE: usize = match parse_usize(option_env!("STORAGE_BLOCK_SIZE")) {
    Some(v) => v,
    None => 512,
};

const fn parse_usize(opt: Option<&str>) -> Option<usize> {
    let Some(s) = opt else { return None };
    let b = s.as_bytes();
    let mut v: usize = 0;
    let mut i = 0;
    while i < b.len() {
        if b[i] < b'0' || b[i] > b'9' {
            return None;
        }
        v = v * 10 + (b[i] - b'0') as usize;
        i += 1;
    }
    Some(v)
}

/// Layer-table magic: 16-bit 0xAE86 signature zero-extended to u32.
pub const LAYER_TABLE_MAGIC: u32 = 0x0000_AE86;
/// Fixed root: LBA 0 / byte offset 0 of the flat blob.
pub const LAYER_TABLE_LBA: u64 = 0;
/// Directory magic: 16-bit 0xAE87 signature zero-extended to u32.
pub const DIR_MAGIC: u32 = 0x0000_AE87;

/// LBA of the tensor directory: immediately after the expert region, derived
/// from the LBA-0 table (never hardcoded — LBA 1 is expert (0,0)'s head, an
/// empirically costly bug: it made the boot probe print the directory magic
/// as "expert00 head", i.e. one expert was being overwritten).
#[inline]
pub const fn dir_lba_after_experts(base0: u64, expert_stride: u64) -> u64 {
    (base0 + NUM_LAYERS as u64 * NUM_EXPERTS_PER_LAYER as u64 * expert_stride)
        / STORAGE_BLOCK_SIZE as u64
}
/// OLMoE-1B-7B geometry (firmware decode path; Qwen profile needs its own).
pub const NUM_HEADS: usize = 16;
pub const NUM_KV_HEADS: usize = 16;
pub const HEAD_DIM: usize = 128;
pub const VOCAB_SIZE: usize = 50304;
pub const MAX_TOKENS: usize = 128;
/// Greedy decode steps per prompt. Prompt ids live in src/prompt.rs (generated
/// by tools/gen_prompt.py) so a run is a rebuild, not an image repack.
pub const DECODE_STEPS: usize = 32;
