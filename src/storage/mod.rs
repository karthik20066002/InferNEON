//! storage/mod.rs — flat-blob block abstraction (backend swap, not redesign).
//!
//! Layout (see docs/STORAGE_LAYOUT.md): LBA 0 holds the layer table
//! [magic u32 = 0x0000AE86 | num_layers u32 | expert_stride u32 | reserved u32]
//! + num_layers × u64 layer_base[]. Expert offset is pure arithmetic:
//!   expert_offset = layer_table[layer_id] + expert_index * expert_stride_bytes
//! where expert_stride_bytes is rounded up at export to STORAGE_BLOCK_SIZE.
//!
//! NOTE (eventual target): real CM4 hardware — PCIe Gen2 x1 ceiling, actual
//! eMMC timing — is NOT handled here yet. virtio-blk now; eMMC/PCIe drivers
//! are stubbed siblings (emmc.rs / pcie.rs). Swapping backends must not
//! change this trait or the offset math.

pub mod emmc;
pub mod pcie;
pub mod virtio;

use crate::config::{LAYER_TABLE_MAGIC, STORAGE_BLOCK_SIZE};

/// Storage failure modes. no_std: plain enum, no formatting.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StorageError {
    /// Probe/negotiation failed (bad magic, wrong device, features rejected).
    Init,
    /// Device did not complete the request in the poll budget.
    Timeout,
    /// Device reported a per-request error status.
    Io,
    /// Bad argument (misaligned buffer, unknown block size pairing).
    BadArg,
}

pub type StorageResult<T> = Result<T, StorageError>;

/// Minimum block-device boundary. All reads/writes are whole blocks.
/// `&mut self`: issuing a request mutates device/queue state.
pub trait BlockStorage {
    /// Read `buf.len()` bytes (multiple of STORAGE_BLOCK_SIZE) starting at
    /// block `lba` into `buf`.
    fn read_blocks(&mut self, lba: u64, buf: &mut [u8]) -> StorageResult<()>;
    /// TODO: only needed for future write-back; stub.
    fn write_blocks(&mut self, lba: u64, buf: &[u8]) -> StorageResult<()>;
    /// Total capacity in STORAGE_BLOCK_SIZE blocks.
    fn capacity_blocks(&mut self) -> StorageResult<u64>;
}

/// Fixed-offset blob descriptor inside the flat image.
#[derive(Clone, Copy, Debug)]
pub struct Blob {
    pub offset_bytes: u64,
    pub len_bytes: u64,
}

/// Round one expert's exported size up to the medium width. Side effect:
/// every expert start is block-aligned, no runtime padding logic.
pub const fn expert_stride_rounded(expert_bytes: usize, block_size: usize) -> usize {
    (expert_bytes + block_size - 1) / block_size * block_size
}

/// O(1) expert lookup after one layer-table indirection.
pub const fn expert_offset(
    layer_base: u64,
    expert_stride_bytes: u64,
    expert_index: u32,
) -> u64 {
    layer_base + expert_index as u64 * expert_stride_bytes
}

/// Validate the LBA-0 header. Returns (num_layers, expert_stride) on magic match.
/// Called once at boot after the first block read; halt on mismatch.
pub fn check_layer_table_header(magic: u32, num_layers: u32, expert_stride: u32) -> Option<(u32, u32)> {
    if magic != LAYER_TABLE_MAGIC {
        return None;
    }
    if expert_stride as usize % STORAGE_BLOCK_SIZE != 0 {
        return None;
    }
    Some((num_layers, expert_stride))
}
