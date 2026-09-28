//! sync.rs — MP-safe primitives for the N-core future. Single core for v1.
//!
//! TODO: real spin Mutex / Barrier / Once. Stubs only; callers must not rely
//! on blocking behavior yet.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicUsize, Ordering};

/// Park the core in low-power wait state (WFE). Used for secondary-core
/// parking and stub idle loops.
#[inline(always)]
pub fn park() {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("wfe", options(nomem, nostack, preserves_flags));
    }
    #[cfg(not(target_arch = "aarch64"))]
    core::hint::spin_loop();
}

/// Spin barrier: all `parties` cores must call `wait` before any proceeds.
/// v1: constructed with 1 party (no-op). N-core: parties = active core count.
pub struct Barrier {
    parties: AtomicUsize,
    arrived: AtomicUsize,
    // TODO: generation counter to make the barrier reusable across layers.
}

impl Barrier {
    pub const fn new(parties: usize) -> Self {
        Self {
            parties: AtomicUsize::new(parties),
            arrived: AtomicUsize::new(0),
        }
    }

    /// TODO: spin on `arrived` with `wfe`/`sev`; needs DSB/memory ordering review.
    pub fn wait(&self) {
        todo!("sync::Barrier::wait")
    }

    pub fn parties(&self) -> usize {
        self.parties.load(Ordering::Relaxed)
    }
}

/// Minimal spin mutex over `UnsafeCell`. IRQs must be masked by the holder
/// once EL1/DAIF handling lands (see boot.rs TODO).
pub struct SpinMutex<T> {
    locked: AtomicUsize,
    value: UnsafeCell<T>,
}

unsafe impl<T: Send> Sync for SpinMutex<T> {}
unsafe impl<T: Send> Send for SpinMutex<T> {}

impl<T> SpinMutex<T> {
    pub const fn new(value: T) -> Self {
        Self {
            locked: AtomicUsize::new(0),
            value: UnsafeCell::new(value),
        }
    }

    /// TODO: compare_exchange loop + wfe backoff.
    pub fn lock(&self) -> &mut T {
        todo!("sync::SpinMutex::lock")
    }
}
