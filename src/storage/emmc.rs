//! storage/emmc.rs — real eMMC driver boundary. STUB ONLY, do not implement.
//! Eventual CM4 target timing/erase geometry lives here later; the
//! BlockStorage trait + flat-blob math above must not change for it.

use super::{BlockStorage, StorageError, StorageResult};

/// Real eMMC controller stub. Never constructed on QEMU virt.
pub struct EmmcDriver;

impl BlockStorage for EmmcDriver {
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
