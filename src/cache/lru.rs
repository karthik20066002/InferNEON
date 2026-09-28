//! cache/lru.rs — LRU tier: warm / recently loaded experts.

use crate::gate::ExpertId;

/// Fixed-capacity warm set. Fresh fetches land here; promotion drains to MFU.
/// TODO: array + recency order (no alloc).
pub struct LruTier {
    capacity: usize,
    occupancy: usize,
}

impl LruTier {
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
        todo!("cache::lru::contains")
    }
    pub fn insert(&mut self, _expert: ExpertId) {
        todo!("cache::lru::insert")
    }
    pub fn remove(&mut self, _expert: ExpertId) {
        todo!("cache::lru::remove")
    }
    pub fn touch(&mut self, _expert: ExpertId) {
        todo!("cache::lru::touch")
    }
}
