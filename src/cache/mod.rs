//! cache/mod.rs — Step 5: per-selected-expert cache resolve.
//! Two tiers: MFU (hot/promoted) + LRU (warm/recently loaded).
//! See policy.rs for promote/swap/evict rules (full names, no S_e shorthand).

pub mod lru;
pub mod mfu;
pub mod policy;
pub mod stats;

pub use lru::LruTier;
pub use mfu::MfuTier;
#[allow(unused_imports)]
pub use stats::{ExpertStats, StatsTable};

use crate::gate::ExpertId;

/// Where the expert was found (or that it missed both tiers).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CacheResidency {
    MfuHit,
    LruHit,
    Miss,
}

/// Resolve one expert: hit → touch stats; miss → fetch via storage then
/// admit via policy (promote / swap / evict). TODO: wire to storage backend.
pub trait CacheResolve {
    fn resolve(&mut self, expert: ExpertId) -> CacheResidency;
}

/// Combined two-tier cache stub.
pub struct ExpertCache {
    pub mfu: MfuTier,
    pub lru: LruTier,
    pub stats: StatsTable,
}

impl ExpertCache {
    pub const fn new(mfu_capacity: usize, lru_capacity: usize) -> Self {
        Self {
            mfu: MfuTier::new(mfu_capacity),
            lru: LruTier::new(lru_capacity),
            stats: StatsTable::new(),
        }
    }
}

impl CacheResolve for ExpertCache {
    fn resolve(&mut self, _expert: ExpertId) -> CacheResidency {
        todo!("cache::ExpertCache::resolve")
    }
}
