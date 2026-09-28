//! cache/stats.rs — per-expert accounting for every expert ever seen.
//! lifetime_call_count (C): global, never resets.
//! accesses_since_last_use (A): derived from a monotonic access tick minus last_used.
//! last_used: tick of the expert's most recent touch (tie-break: older wins).

use crate::gate::ExpertId;

/// Stats for one expert.
#[derive(Clone, Copy, Debug)]
pub struct ExpertStats {
    /// Lifetime call count. Global, never resets.
    pub lifetime_call_count: u64,
    /// Monotonic tick of the last touch (for age + tie-breaks).
    pub last_used: u64,
}

/// Global table stub. TODO: fixed array indexed by global expert id
/// (layer_id * NUM_EXPERTS_PER_LAYER + expert_index), no hashmap (no alloc).
pub struct StatsTable {
    /// Monotonic access tick, bumped on every touch/resolve.
    now: u64,
}

impl StatsTable {
    pub const fn new() -> Self {
        Self { now: 0 }
    }

    /// Record a touch: bump tick, increment lifetime_call_count, stamp last_used.
    /// TODO: real array update.
    pub fn touch(&mut self, _expert: ExpertId) {
        todo!("cache::stats::touch")
    }

    /// accesses_since_last_use = now - last_used (saturating).
    pub fn accesses_since_last_use(&self, _expert: ExpertId) -> u64 {
        todo!("cache::stats::accesses_since_last_use")
    }

    pub fn lifetime_call_count(&self, _expert: ExpertId) -> u64 {
        todo!("cache::stats::lifetime_call_count")
    }

    pub fn now(&self) -> u64 {
        self.now
    }
}
