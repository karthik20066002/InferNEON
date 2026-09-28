//! cache/mfu.rs — MFU tier: hot / promoted experts.

use crate::gate::ExpertId;

/// Fixed-capacity hot set. Membership changes only via promote/swap/demote.
/// TODO: array + occupancy bitmap (no alloc).
pub struct MfuTier {
    capacity: usize,
    occupancy: usize,
}

impl MfuTier {
    pub const fn new(capacity: usize) -> Self {
        Self { capacity, occupancy: 0 }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn occupancy(&self) -> usize {
        self.occupancy
    }
    pub fn has_room(&self) -> bool {
        self.occupancy < self.capacity
    }

    pub fn contains(&self, _expert: ExpertId) -> bool {
        todo!("cache::mfu::contains")
    }
    pub fn insert(&mut self, _expert: ExpertId) {
        todo!("cache::mfu::insert")
    }
    pub fn remove(&mut self, _expert: ExpertId) {
        todo!("cache::mfu::remove")
    }
}
