//! fetch.rs — weight fetch from the flat blob (no filesystem).
//!
//! Directory at FIXED LBA 1 (see tools/pack_olmoe_full.py): magic2, count,
//! then (tag u32, offset u64, len u64, dtype u32) entries. Tags: 1 embed,
//! 2 routers, 3 norms, 4 attn, 5 head, 6 rope_cos, 7 rope_sin,
//! 8 qk_norms (per-layer q,k RMSNorm weights).
//! Staging buffers are u16 arrays (alignment-safe for bf16); byte views for
//! read_blocks go through one unsafe cast each, lengths always exact.
//! Everything converts bf16->f32 on the way into compute (exact shift).

use crate::config::*;
use crate::nnmath::bf16_to_f32;
use crate::storage::BlockStorage;

pub const TAG_EMBED: u32 = 1;
pub const TAG_ROUTERS: u32 = 2;
pub const TAG_NORMS: u32 = 3;
pub const TAG_ATTN: u32 = 4;
pub const TAG_HEAD: u32 = 5;
pub const TAG_ROPE_COS: u32 = 6;
pub const TAG_ROPE_SIN: u32 = 7;

const DTYPE_BF16: u32 = 0;
const _DTYPE_FP32: u32 = 1;

/// One directory entry. Populated once at boot by [`read_dir_at`].
#[derive(Clone, Copy, Debug)]
pub struct DirEntry {
    pub tag: u32,
    pub offset: u64,
    pub len: u64,
    pub dtype: u32,
}

const DIR_MAX_ENTRIES: usize = 12;
static mut DIR: [DirEntry; DIR_MAX_ENTRIES] = [DirEntry { tag: 0, offset: 0, len: 0, dtype: 0 }; DIR_MAX_ENTRIES];
static mut DIR_LEN: usize = 0;

/// Read and validate the directory at `lba` (derived, not hardcoded — see
/// config::dir_lba_after_experts). Returns entry count.
pub fn read_dir_at<B: BlockStorage>(blk: &mut B, lba: u64) -> Result<usize, ()> {
    let mut raw = [0u8; 512];
    blk.read_blocks(lba, &mut raw).map_err(|_| ())?;
    let magic = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
    if magic != DIR_MAGIC {
        return Err(());
    }
    let count = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]) as usize;
    if count == 0 || count > DIR_MAX_ENTRIES {
        return Err(());
    }
    unsafe {
        for i in 0..count {
            // Stride 28, not 24: the packer writes '<IQQI4x' (tag 4 +
            // offset 8 + len 8 + dtype 4 + 4 pad). Reading 24-byte entries
            // desynchronises the field offsets after the first one, which is
            // why every tag resolved to a garbage offset here.
            let o = 8 + i * 28;
            DIR[i] = DirEntry {
                tag: u32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]]),
                offset: u64::from_le_bytes([
                    raw[o + 4], raw[o + 5], raw[o + 6], raw[o + 7], raw[o + 8], raw[o + 9],
                    raw[o + 10], raw[o + 11],
                ]),
                len: u64::from_le_bytes([
                    raw[o + 12], raw[o + 13], raw[o + 14], raw[o + 15], raw[o + 16], raw[o + 17],
                    raw[o + 18], raw[o + 19],
                ]),
                dtype: u32::from_le_bytes([raw[o + 20], raw[o + 21], raw[o + 22], raw[o + 23]]),
            };
        }
        DIR_LEN = count;
    }
    Ok(count)
}

fn lookup(tag: u32) -> (u64, u64, u32) {
    unsafe {
        for i in 0..DIR_LEN {
            if DIR[i].tag == tag {
                return (DIR[i].offset, DIR[i].len, DIR[i].dtype);
            }
        }
    }
    (0, 0, 0xFFFF_FFFF)
}

/// Read `len` bytes at absolute byte `off` into `buf` (512-multiple length).
fn read_range<B: BlockStorage>(blk: &mut B, off: u64, buf: &mut [u8]) -> Result<(), ()> {
    if off % 512 != 0 || buf.len() % 512 != 0 {
        return Err(());
    }
    blk.read_blocks(off / 512, buf).map_err(|_| ())
}

/// Byte view of a u16 staging buffer for block reads.
fn as_bytes(buf: &mut [u16]) -> &mut [u8] {
    unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, buf.len() * 2) }
}

// ---------------------------------------------------------------------------
// Static staging (linker-sized; OOM is a link error, never runtime).
// Worst case: 12.6 + 33.6 + 25.7 + 8(KV f32) + 4 + 1 ≈ 86 MiB < 128 MiB.
// ---------------------------------------------------------------------------

/// One expert, raw bf16 (gate 2M + up 2M + down 4M u16).
pub static mut EXPERT_STAGE: [u16; 3 * 2048 * 1024] = [0; 3 * 2048 * 1024];
/// One layer's attn q,k,v,o raw bf16 (4 x 2048x2048 u16).
pub static mut ATTN_STAGE: [u16; 4 * 2048 * 2048] = [0; 4 * 2048 * 2048];
/// One head shard raw bf16 (VOCAB/8 rows x 2048).
pub const HEAD_SHARDS: usize = 8;
pub const HEAD_SHARD_ROWS: usize = VOCAB_SIZE / HEAD_SHARDS;
pub static mut HEAD_STAGE: [u16; HEAD_SHARD_ROWS * 2048] = [0; HEAD_SHARD_ROWS * 2048];
/// All routers pinned at boot, converted f32 [16, 64, 2048].
pub static mut ROUTERS_F32: [f32; 16 * 64 * 2048] = [0.0; 16 * 64 * 2048];
/// All norms pinned at boot, converted f32 [33, 2048].
pub static mut NORMS_F32: [f32; 33 * 2048] = [0.0; 33 * 2048];
/// Scratch for pinned-router conversion (256 KiB; static, not stack).
static mut ROUTER_RAW: [u16; 64 * 2048] = [0; 64 * 2048];
/// Scratch for pinned-norm conversion (4 KiB).
static mut NORM_RAW: [u16; 2048] = [0; 2048];
/// Scratch for embed-row conversion (4 KiB).
static mut EMB_RAW: [u16; 2048] = [0; 2048];
/// Per-layer q/k norms pinned at boot, f32 [32, 2048] (q0,k0,q1,k1...).
pub static mut QKNORM_F32: [f32; 32 * 2048] = [0.0; 32 * 2048];
/// RoPE tables pinned at boot, fp32 direct copy [16, 64] each.
pub static mut ROPE_COS: [f32; MAX_TOKENS * 64] = [0.0; MAX_TOKENS * 64];
pub static mut ROPE_SIN: [f32; MAX_TOKENS * 64] = [0.0; MAX_TOKENS * 64];

/// Fused bf16 dot: sum(bf16_to_f32(row[i]) * x[i]). Single pass, no scratch.
pub fn dot_bf16(row: &[u16], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..x.len() {
        acc += bf16_to_f32(row[i]) * x[i];
    }
    acc
}

/// Fetch one expert's gate/up/down into EXPERT_STAGE. Returns (gate, up, down)
/// as u16 slices into staging (caller converts per row via dot_bf16).
pub fn fetch_expert<B: BlockStorage>(
    blk: &mut B,
    layer_base: u64,
    expert_stride: u64,
    expert: u32,
) -> Result<(), ()> {
    let off = layer_base + expert as u64 * expert_stride;
    unsafe {
        let buf = as_bytes(&mut EXPERT_STAGE);
        // EXPERT_BYTES is 512-aligned by packer invariant; read in 1MiB chunks
        // to keep single requests modest (virtio handles any 512-multiple).
        let total = EXPERT_BYTES;
        let mut done = 0usize;
        while done < total {
            let n = core::cmp::min(1 << 20, total - done);
            read_range(blk, off + done as u64, &mut buf[done..done + n])?;
            done += n;
        }
    }
    Ok(())
}

/// Fetch one layer's q,k,v,o into ATTN_STAGE (each 2048x2048 bf16, in order).
pub fn fetch_attn<B: BlockStorage>(blk: &mut B, layer: u32) -> Result<(), ()> {
    let (off, len, dtype) = lookup(TAG_ATTN);
    if dtype != DTYPE_BF16 {
        return Err(());
    }
    let per_layer = (4 * 2048 * 2048 * 2) as u64;
    assert_eq!(len, per_layer * 16);
    unsafe {
        let buf = as_bytes(&mut ATTN_STAGE);
        let total = per_layer as usize;
        let mut done = 0usize;
        while done < total {
            let n = core::cmp::min(1 << 20, total - done);
            read_range(blk, off + layer as u64 * per_layer + done as u64, &mut buf[done..done + n])?;
            done += n;
        }
    }
    Ok(())
}

/// Fetch one head shard (rows shard*6288..) into HEAD_STAGE.
pub fn fetch_head_shard<B: BlockStorage>(blk: &mut B, shard: usize) -> Result<(), ()> {
    let (off, len, dtype) = lookup(TAG_HEAD);
    if dtype != DTYPE_BF16 {
        return Err(());
    }
    assert_eq!(len as usize, VOCAB_SIZE * 2048 * 2);
    let rows = HEAD_SHARD_ROWS;
    let bytes = rows * 2048 * 2;
    unsafe {
        let buf = as_bytes(&mut HEAD_STAGE);
        debug_assert_eq!(buf.len(), bytes);
        let mut done = 0usize;
        while done < bytes {
            let n = core::cmp::min(1 << 20, bytes - done);
            read_range(
                blk,
                off + shard as u64 * bytes as u64 + done as u64,
                &mut buf[done..done + n],
            )?;
            done += n;
        }
    }
    Ok(())
}

/// Fetch one embed row into `out` as f32 (row = token id).
pub fn fetch_embed_row<B: BlockStorage>(blk: &mut B, id: u32, out: &mut [f32]) -> Result<(), ()> {
    let (off, len, dtype) = lookup(TAG_EMBED);
    if dtype != DTYPE_BF16 {
        return Err(());
    }
    assert_eq!(len as usize, VOCAB_SIZE * 2048 * 2);
    assert!((id as usize) < VOCAB_SIZE && out.len() == 2048);
    let raw = unsafe { &mut EMB_RAW };
    let bytes = as_bytes(&mut raw[..]);
    debug_assert_eq!(bytes.len(), 4096);
    read_range(blk, off + id as u64 * 4096, bytes)?;
    for i in 0..2048 {
        out[i] = bf16_to_f32(raw[i]);
    }
    Ok(())
}

/// Pin routers + norms + rope tables at boot (small, reused every token).
pub fn pin_resident<B: BlockStorage>(blk: &mut B) -> Result<(), ()> {
    unsafe {
        // routers: 16 x [64,2048] bf16 -> f32
        let (off, len, dtype) = lookup(TAG_ROUTERS);
        if dtype != DTYPE_BF16 || len as usize != 16 * 64 * 2048 * 2 {
            return Err(());
        }
        let raw = unsafe { &mut ROUTER_RAW };
        for li in 0..16usize {
            let bytes = as_bytes(&mut raw[..]);
            read_range(blk, off + (li * 64 * 2048 * 2) as u64, bytes)?;
            for i in 0..64 * 2048 {
                ROUTERS_F32[li * 64 * 2048 + i] = bf16_to_f32(raw[i]);
            }
        }
        // norms: 33 x [2048] as stored (packer says BF16) -> f32
        let (off, len, dtype) = lookup(TAG_NORMS);
        if dtype != DTYPE_BF16 || len as usize != 33 * 2048 * 2 {
            return Err(());
        }
        let nraw = unsafe { &mut NORM_RAW };
        for ni in 0..33usize {
            let bytes = as_bytes(&mut nraw[..]);
            read_range(blk, off + (ni * 2048 * 2) as u64, bytes)?;
            for i in 0..2048 {
                NORMS_F32[ni * 2048 + i] = bf16_to_f32(nraw[i]);
            }
        }
        // q/k norms per layer (tag 8): 32 x [2048] bf16, rows q0,k0,q1,k1...
        let (off, len, dtype) = lookup(8);
        if dtype != DTYPE_BF16 || len as usize != 32 * 2048 * 2 {
            return Err(());
        }
        let nraw = unsafe { &mut NORM_RAW };
        for ni in 0..32usize {
            let bytes = as_bytes(&mut nraw[..]);
            read_range(blk, off + (ni * 2048 * 2) as u64, bytes)?;
            for i in 0..2048 {
                unsafe { QKNORM_F32[ni * 2048 + i] = bf16_to_f32(nraw[i]); }
            }
        }
        // rope tables: fp32 direct copy
        let (off, len, dtype) = lookup(TAG_ROPE_COS);
        if dtype != 1 || len as usize != MAX_TOKENS * 64 * 4 {
            return Err(());
        }
        let cos_b = core::slice::from_raw_parts_mut(ROPE_COS.as_mut_ptr() as *mut u8, len as usize);
        read_range(blk, off, cos_b)?;
        let (off, len, dtype) = lookup(TAG_ROPE_SIN);
        if dtype != 1 || len as usize != MAX_TOKENS * 64 * 4 {
            return Err(());
        }
        let sin_b = core::slice::from_raw_parts_mut(ROPE_SIN.as_mut_ptr() as *mut u8, len as usize);
        read_range(blk, off, sin_b)?;
    }
    Ok(())
}
