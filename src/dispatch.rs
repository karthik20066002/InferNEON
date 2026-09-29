//! dispatch.rs — Step 4: the k experts are an UNORDERED SET behind a barrier.
//! Core assignment hangs off this step (1 core today, N with multicore).

use crate::config::MAX_K;
use crate::gate::ExpertId;

/// Deduplicated unordered set of selected experts. Order MUST NOT be relied on.
/// Qwen profile (HAS_SHARED_EXPERT): the shared expert joins every dispatch
/// implicitly alongside the top-k routed set — TODO in from_top_k.
#[derive(Clone, Copy, Debug)]
pub struct ExpertSet {
    ids: [Option<ExpertId>; MAX_K],
    len: usize,
}

impl ExpertSet {
    pub const fn empty() -> Self {
        Self { ids: [None; MAX_K], len: 0 }
    }

    /// Build from `gating` output prefix `k`. Deduplicates defensively.
    pub fn from_top_k(ids: &[ExpertId], k: usize) -> Self {
        assert!(k <= MAX_K);
        let mut set = Self::empty();
        for id in ids.iter().take(k) {
            if set.len >= MAX_K {
                break;
            }
            let mut dup = false;
            for slot in set.ids.iter().take(set.len) {
                if *slot == Some(*id) {
                    dup = true;
                    break;
                }
            }
            if !dup {
                set.ids[set.len] = Some(*id);
                set.len += 1;
            }
        }
        set
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = ExpertId> + '_ {
        self.ids.iter().filter_map(|slot| *slot)
    }
}

/// Bitmask of cores assigned to this dispatch (bit i = core i). v1: bit 0 only.
#[derive(Clone, Copy, Debug)]
pub struct CoreMask(pub u64);

/// Decide which core(s) run each expert in the set.
/// v1 single-core: everything on core 0. N-core sharding is future work.
pub fn assign_cores(_set: &ExpertSet) -> CoreMask {
    CoreMask(1)
}
