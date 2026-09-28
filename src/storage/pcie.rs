//! storage/pcie.rs — real PCIe/NVMe (M.2) driver boundary. STUB ONLY, do not implement.
//! Eventual CM4 target is PCIe Gen2 x1 ceiling; that constraint lives here
//! later. Backend swap only — trait + layout above stay fixed.

use super::{BlockStorage, StorageError, StorageResult};

/// Real PCIe/NVMe controller stub. Never constructed on QEMU virt.
pub struct PcieNvmeDriver;

impl BlockStorage for PcieNvmeDriver {
    fn read_blocks(&mut self, _lba: u64, _buf: &mut [u8]) -> StorageResult<()> {
        Err(StorageError::Init)
    }
    fn write_blocks(&mut self, _lba: u64, _buf: &[u8]) -> StorageResult<()> {
        Err(StorageError::Init)
    }
    fn capacity_blocks(&mut self) -> StorageResult<u64> {
        Err(StorageError::Init)
    }
}
