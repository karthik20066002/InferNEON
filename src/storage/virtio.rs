//! storage/virtio.rs — QEMU virtio-blk driver over virtio-mmio (modern v2).
//!
//! Probe → feature negotiate (VERSION_1 only) → one 16-entry virtqueue →
//! polled single-sector reads through a 512 B staging buffer. Single core,
//! single outstanding request (N-core TODO: one queue per core).
//! Sector width is the native 512 B; STORAGE_BLOCK_SIZE ≥ 512 must be a
//! multiple of it (STORAGE_BLOCK_SIZE=512 → 1:1).

use super::{BlockStorage, StorageError, StorageResult};
use crate::addr::VIRTIO_MMIO_BASE;
use crate::config::STORAGE_BLOCK_SIZE;
use core::sync::atomic::{fence, Ordering};

const NATIVE_SECTOR: usize = 512;
const QUEUE_LEN: usize = 16;
const POLL_BUDGET: u32 = 20_000_000;

// virtio-mmio register offsets. Modern (v2) names; legacy (v1) reuses
// MAGIC/VERSION/DEVICE_ID/FEATURES/QUEUE_SEL/NUM_MAX/NUM/NOTIFY/INT/STATUS,
// with the legacy queue wired via GUEST_PAGE_SIZE/QUEUE_ALIGN/QUEUE_PFN.
const R_MAGIC: usize = 0x000;
const R_VERSION: usize = 0x004;
const R_DEVICE_ID: usize = 0x008;
const R_DEVICE_FEAT: usize = 0x010;
const R_DEVICE_FEAT_SEL: usize = 0x014;
const R_DRIVER_FEAT: usize = 0x020;
const R_DRIVER_FEAT_SEL: usize = 0x024;
const R_GUEST_PAGE_SIZE: usize = 0x028; // legacy only
const R_QUEUE_SEL: usize = 0x030;
const R_QUEUE_NUM_MAX: usize = 0x034;
const R_QUEUE_NUM: usize = 0x038;
const R_QUEUE_ALIGN: usize = 0x03c; // legacy only
const R_QUEUE_PFN: usize = 0x040; // legacy only
const R_QUEUE_READY: usize = 0x044; // modern only
const R_QUEUE_NOTIFY: usize = 0x050;
const R_INT_STATUS: usize = 0x060;
const R_INT_ACK: usize = 0x064;
const R_STATUS: usize = 0x070;
const R_QUEUE_DESC_LO: usize = 0x080; // modern only
const R_QUEUE_DESC_HI: usize = 0x084;
const R_QUEUE_AVAIL_LO: usize = 0x090;
const R_QUEUE_AVAIL_HI: usize = 0x094;
const R_QUEUE_USED_LO: usize = 0x0a0;
const R_QUEUE_USED_HI: usize = 0x0a4;
const R_CONFIG_CAP: usize = 0x100; // u64 capacity in native sectors

const MAGIC_VALUE: u32 = 0x74726976; // "triv"
const VERSION_MODERN: u32 = 2;
const VERSION_LEGACY: u32 = 1;
const DEVICE_ID_BLOCK: u32 = 2;

const ST_ACKNOWLEDGE: u8 = 1;
const ST_DRIVER: u8 = 2;
const ST_DRIVER_OK: u8 = 4;
const ST_FEATURES_OK: u8 = 8;

const DESC_F_NEXT: u16 = 1;
const DESC_F_WRITE: u16 = 2; // device writes (driver reads)

const REQ_TYPE_IN: u32 = 0;
const REQ_STATUS_OK: u8 = 0;
// VIRTIO_F_VERSION_1 lives in feature word 1 (bits 32..63).
const F_VERSION_1_WORD1: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
struct Desc {
    addr: u64,
    len: u32,
    flags: u16,
    next: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Avail {
    flags: u16,
    idx: u16,
    ring: [u16; QUEUE_LEN],
    used_event: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct UsedElem {
    id: u32,
    len: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Used {
    flags: u16,
    idx: u16,
    ring: [UsedElem; QUEUE_LEN],
    avail_event: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlkReq {
    req_type: u32,
    reserved: u32,
    sector: u64,
}

// Queue + request memory. Statics (BSS): identity-mapped RAM, visible to the
// device. Layout is legacy-compatible AND modern-usable: descriptors at page
// base, avail ring immediately after, used ring on the next 4096 boundary
// (legacy QueueAlign=4096). Modern paths just take the three addresses.
const QUEUE_PAGE_SIZE: u32 = 4096;
const DESC_BYTES: usize = QUEUE_LEN * 16;
const AVAIL_OFF: usize = DESC_BYTES; // 256
const USED_OFF: usize = QUEUE_PAGE_SIZE as usize; // 4096

#[repr(align(4096))]
struct QueuePages([u8; 8192]);

static mut QUEUE_PAGES: QueuePages = QueuePages([0; 8192]);
static mut REQ: BlkReq = BlkReq { req_type: 0, reserved: 0, sector: 0 };
static mut DATA: [u8; NATIVE_SECTOR] = [0; NATIVE_SECTOR];
static mut REQ_STATUS: u8 = 0;

/// Pointers to (desc table, avail ring, used ring) inside QUEUE_PAGES.
fn queue_ptrs() -> (*mut Desc, *mut Avail, *mut Used) {
    unsafe {
        let base = core::ptr::addr_of_mut!(QUEUE_PAGES.0) as *mut u8;
        (
            base.add(0) as *mut Desc,
            base.add(AVAIL_OFF) as *mut Avail,
            base.add(USED_OFF) as *mut Used,
        )
    }
}

#[inline]
fn mmio_r32(base: usize, off: usize) -> u32 {
    unsafe { core::ptr::read_volatile((base + off) as *const u32) }
}

#[inline]
fn mmio_r64(base: usize, off: usize) -> u64 {
    unsafe { core::ptr::read_volatile((base + off) as *const u64) }
}

#[inline]
fn mmio_w32(base: usize, off: usize, v: u32) {
    unsafe { core::ptr::write_volatile((base + off) as *mut u32, v) };
}

// NOTE: STATUS (like every virtio-mmio control register here) is written
// 32 bits wide — QEMU ignores sub-word writes to it.

/// virtio-blk device over virtio-mmio. Construct with `new()` (scans the
/// board's transports for a block device), then `init()`.
pub struct VirtioBlk {
    base: usize,
    ready: bool,
    avail_idx: u16,
    last_used_idx: u16,
}

impl VirtioBlk {
    /// Scan transports at VIRTIO_MMIO_BASE + i*0x200 for magic + block id.
    /// Falls back to transport 0 (init will report the failure precisely).
    pub fn new() -> Self {
        let mut base = VIRTIO_MMIO_BASE;
        for i in 0..32u32 {
            let b = VIRTIO_MMIO_BASE + i as usize * 0x200;
            if mmio_r32(b, R_MAGIC) == MAGIC_VALUE
                && mmio_r32(b, R_DEVICE_ID) == DEVICE_ID_BLOCK
            {
                base = b;
                break;
            }
        }
        Self { base, ready: false, avail_idx: 0, last_used_idx: 0 }
    }

    /// Probe the transport and negotiate. Safe to call once; re-init resets.
    /// Accepts modern (v2) and legacy (v1) transports — QEMU's `virt` board
    /// defaults to legacy unless `-global virtio-mmio.force-legacy=off`.
    pub fn init(&mut self) -> StorageResult<()> {
        if mmio_r32(self.base, R_MAGIC) != MAGIC_VALUE {
            crate::uart::print("storage: no mmio magic\n");
            return Err(StorageError::Init);
        }
        let vers = mmio_r32(self.base, R_VERSION);
        if vers != VERSION_MODERN && vers != VERSION_LEGACY {
            crate::uart::print("storage: bad mmio version=");
            crate::uart::print_hex_u32(vers);
            crate::uart::print("\n");
            return Err(StorageError::Init);
        }
        let devid = mmio_r32(self.base, R_DEVICE_ID);
        if devid != DEVICE_ID_BLOCK {
            crate::uart::print("storage: not blk, id=");
            crate::uart::print_hex_u32(devid);
            crate::uart::print("\n");
            return Err(StorageError::Init);
        }
        // Reset, then ACKNOWLEDGE | DRIVER.
        mmio_w32(self.base, R_STATUS, 0);
        fence(Ordering::SeqCst);
        mmio_w32(self.base, R_STATUS, (ST_ACKNOWLEDGE | ST_DRIVER) as u32);
        if vers == VERSION_MODERN {
            self.setup_modern()?;
        } else {
            self.setup_legacy()?;
        }
        let status = mmio_r32(self.base, R_STATUS);
        mmio_w32(self.base, R_STATUS, status | ST_DRIVER_OK as u32);
        self.ready = true;
        self.avail_idx = 0;
        self.last_used_idx = 0;
        Ok(())
    }

    /// Modern (v2) feature + queue setup. Negotiates VERSION_1 only.
    fn setup_modern(&mut self) -> StorageResult<()> {
        mmio_w32(self.base, R_DEVICE_FEAT_SEL, 1);
        let _offered_hi = mmio_r32(self.base, R_DEVICE_FEAT);
        mmio_w32(self.base, R_DRIVER_FEAT_SEL, 1);
        mmio_w32(self.base, R_DRIVER_FEAT, F_VERSION_1_WORD1);
        mmio_w32(self.base, R_DRIVER_FEAT_SEL, 0);
        mmio_w32(self.base, R_DRIVER_FEAT, 0);
        let mut status = mmio_r32(self.base, R_STATUS);
        status |= ST_FEATURES_OK as u32;
        mmio_w32(self.base, R_STATUS, status);
        fence(Ordering::SeqCst);
        if mmio_r32(self.base, R_STATUS) & ST_FEATURES_OK as u32 == 0 {
            crate::uart::print("storage: features rejected\n");
            return Err(StorageError::Init);
        }
        mmio_w32(self.base, R_QUEUE_SEL, 0);
        let qmax = mmio_r32(self.base, R_QUEUE_NUM_MAX);
        if qmax < QUEUE_LEN as u32 {
            crate::uart::print("storage: qmax=");
            crate::uart::print_hex_u32(qmax);
            crate::uart::print("\n");
            return Err(StorageError::Init);
        }
        mmio_w32(self.base, R_QUEUE_NUM, QUEUE_LEN as u32);
        let (desc, avail, used) = queue_ptrs();
        self.zero_queue();
        mmio_w32(self.base, R_QUEUE_DESC_LO, desc as u32);
        mmio_w32(self.base, R_QUEUE_DESC_HI, ((desc as u64) >> 32) as u32);
        mmio_w32(self.base, R_QUEUE_AVAIL_LO, avail as u32);
        mmio_w32(self.base, R_QUEUE_AVAIL_HI, ((avail as u64) >> 32) as u32);
        mmio_w32(self.base, R_QUEUE_USED_LO, used as u32);
        mmio_w32(self.base, R_QUEUE_USED_HI, ((used as u64) >> 32) as u32);
        mmio_w32(self.base, R_QUEUE_READY, 1);
        fence(Ordering::SeqCst);
        Ok(())
    }

    /// Legacy (v1) feature + queue setup. Single 32-bit feature word;
    /// basic IN reads need no transport features, so ack 0. Queue memory is
    /// one 4096-aligned region pointed at by QUEUE_PFN.
    fn setup_legacy(&mut self) -> StorageResult<()> {
        let _offered = mmio_r32(self.base, R_DEVICE_FEAT);
        mmio_w32(self.base, R_DRIVER_FEAT, 0);
        let mut status = mmio_r32(self.base, R_STATUS);
        status |= ST_FEATURES_OK as u32;
        mmio_w32(self.base, R_STATUS, status);
        fence(Ordering::SeqCst);
        if mmio_r32(self.base, R_STATUS) & ST_FEATURES_OK as u32 == 0 {
            crate::uart::print("storage: features rejected\n");
            return Err(StorageError::Init);
        }
        mmio_w32(self.base, R_GUEST_PAGE_SIZE, QUEUE_PAGE_SIZE);
        mmio_w32(self.base, R_QUEUE_SEL, 0);
        let qmax = mmio_r32(self.base, R_QUEUE_NUM_MAX);
        if qmax < QUEUE_LEN as u32 {
            crate::uart::print("storage: qmax=");
            crate::uart::print_hex_u32(qmax);
            crate::uart::print("\n");
            return Err(StorageError::Init);
        }
        mmio_w32(self.base, R_QUEUE_NUM, QUEUE_LEN as u32);
        mmio_w32(self.base, R_QUEUE_ALIGN, QUEUE_PAGE_SIZE);
        let (desc, _avail, _used) = queue_ptrs();
        self.zero_queue();
        mmio_w32(
            self.base,
            R_QUEUE_PFN,
            (desc as u64 / QUEUE_PAGE_SIZE as u64) as u32,
        );
        fence(Ordering::SeqCst);
        Ok(())
    }

    /// Zero the shared queue pages (fresh idx state for both paths).
    fn zero_queue(&mut self) {
        unsafe {
            let base = core::ptr::addr_of_mut!(QUEUE_PAGES.0) as *mut u8;
            core::ptr::write_bytes(base, 0, 8192);
        }
    }

    fn ensure_ready(&mut self) -> StorageResult<()> {
        if self.ready {
            return Ok(());
        }
        self.init()
    }

    /// Single native-sector IN transfer into the staging buffer.
    /// Queue mechanics are identical for modern and legacy transports.
    fn transfer_in(&mut self, sector: u64) -> StorageResult<()> {
        unsafe {
            REQ = BlkReq { req_type: REQ_TYPE_IN, reserved: 0, sector };
            REQ_STATUS = 0xff;
            let req_addr = core::ptr::addr_of!(REQ) as u64;
            let data_addr = core::ptr::addr_of!(DATA) as u64;
            let status_addr = core::ptr::addr_of!(REQ_STATUS) as u64;
            let (desc_ptr, avail_ptr, used_ptr) = queue_ptrs();
            let desc = core::slice::from_raw_parts_mut(desc_ptr, QUEUE_LEN);
            desc[0] = Desc { addr: req_addr, len: 16, flags: DESC_F_NEXT, next: 1 };
            desc[1] = Desc {
                addr: data_addr,
                len: NATIVE_SECTOR as u32,
                flags: DESC_F_NEXT | DESC_F_WRITE,
                next: 2,
            };
            desc[2] = Desc { addr: status_addr, len: 1, flags: DESC_F_WRITE, next: 0 };
            let avail = &mut *avail_ptr;
            avail.ring[(self.avail_idx as usize) % QUEUE_LEN] = 0;
            fence(Ordering::SeqCst);
            self.avail_idx = self.avail_idx.wrapping_add(1);
            avail.idx = self.avail_idx;
            fence(Ordering::SeqCst);
            mmio_w32(self.base, R_QUEUE_NOTIFY, 0);
            // Poll the used ring (no interrupts in v1).
            let mut budget = POLL_BUDGET;
            loop {
                fence(Ordering::SeqCst);
                let used_idx = (&*used_ptr).idx;
                if used_idx != self.last_used_idx {
                    break;
                }
                if budget == 0 {
                    return Err(StorageError::Timeout);
                }
                budget -= 1;
                core::hint::spin_loop();
            }
            fence(Ordering::SeqCst);
            let used = &*used_ptr;
            if used.ring[(self.last_used_idx as usize) % QUEUE_LEN].id != 0 {
                return Err(StorageError::Io);
            }
            self.last_used_idx = self.last_used_idx.wrapping_add(1);
            // Ack the used-buffer notification.
            let _ = mmio_r32(self.base, R_INT_STATUS);
            mmio_w32(self.base, R_INT_ACK, 0x1);
            if REQ_STATUS != REQ_STATUS_OK {
                return Err(StorageError::Io);
            }
        }
        Ok(())
    }

    /// Native-sector read into a caller buffer (staging + copy).
    fn read_native(&mut self, sector: u64, out: &mut [u8; NATIVE_SECTOR]) -> StorageResult<()> {
        self.transfer_in(sector)?;
        unsafe {
            out.copy_from_slice(&DATA);
        }
        Ok(())
    }
}

impl BlockStorage for VirtioBlk {
    fn read_blocks(&mut self, lba: u64, buf: &mut [u8]) -> StorageResult<()> {
        if STORAGE_BLOCK_SIZE % NATIVE_SECTOR != 0 || buf.len() % STORAGE_BLOCK_SIZE != 0 {
            return Err(StorageError::BadArg);
        }
        self.ensure_ready()?;
        let sectors_per_block = STORAGE_BLOCK_SIZE / NATIVE_SECTOR;
        let mut sector = lba * sectors_per_block as u64;
        for chunk in buf.chunks_mut(STORAGE_BLOCK_SIZE) {
            for s in 0..sectors_per_block {
                let mut tmp = [0u8; NATIVE_SECTOR];
                self.read_native(sector, &mut tmp)?;
                chunk[s * NATIVE_SECTOR..(s + 1) * NATIVE_SECTOR].copy_from_slice(&tmp);
                sector += 1;
            }
        }
        Ok(())
    }

    fn write_blocks(&mut self, _lba: u64, _buf: &[u8]) -> StorageResult<()> {
        // No write path in v1 (weights are read-only at runtime).
        Err(StorageError::BadArg)
    }

    fn capacity_blocks(&mut self) -> StorageResult<u64> {
        self.ensure_ready()?;
        let native_sectors = mmio_r64(self.base, R_CONFIG_CAP);
        Ok(native_sectors / (STORAGE_BLOCK_SIZE / NATIVE_SECTOR) as u64)
    }
}
