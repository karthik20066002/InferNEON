//! cache/policy.rs — PROMOTE / SWAP / EVICT as pure-function signatures.
//! Full names throughout (no S_e / C_e shorthand in the API).
//! Bodies are TODO stubs; formulas live in doc comments so the contract is
//! reviewable before any implementation lands.
//!
//! PROMOTE gate: occ(LRU) >= 3/4*cap(LRU) AND MFU has room.
//! heat_score = lifetime_call_count / max(accesses_since_last_use, 1).
//! promotion_candidate = argmax over LRU of heat_score. Tie: older last_used wins.
//! coldest_mfu_resident = argmin over MFU of heat_score. Tie: older last_used wins.
//! SWAP: both tiers full AND candidate_heat > coldest_heat → candidate → MFU,
//!   coldest demotes to LRU. Same score/tiebreak as promote.
//! EVICT (fallback: both full, swap lost): drop LRU's own weakest by
//!   log-scaled retention, NOT the heat formula:
//!   frequency_score = (log(1+C)-log(1+C_min)) / (log(1+C_max)-log(1+C_min))
//!     with C_min/C_max over CURRENT LRU residents only
//!     (degenerate C_max==C_min → frequency_score = 1 for all; TODO: lock).
//!   recency_score = 1 / (1 + accesses_since_last_use/96)
//!   retention_score = 0.25*frequency_score + 0.75*recency_score
//!   evict argmin retention_score over LRU.

use crate::gate::ExpertId;

/// heat_score = lifetime_call_count / max(accesses_since_last_use, 1).
pub fn heat_score(lifetime_call_count: u64, accesses_since_last_use: u64) -> u64 {
    let _ = (lifetime_call_count, accesses_since_last_use);
    todo!("cache::policy::heat_score")
}

/// PROMOTE gate: lru_occupancy >= 3/4*lru_capacity AND mfu has room.
/// Integer-safe: `lru_occupancy * 4 >= lru_capacity * 3`.
pub fn should_promote(
    lru_occupancy: usize,
    lru_capacity: usize,
    mfu_occupancy: usize,
    mfu_capacity: usize,
) -> bool {
    let _ = (lru_occupancy, lru_capacity, mfu_occupancy, mfu_capacity);
    todo!("cache::policy::should_promote")
}

/// Hottest LRU resident (argmax heat_score, older last_used breaks ties).
pub fn promotion_candidate(_lru_residents: &[ExpertId]) -> Option<ExpertId> {
    todo!("cache::policy::promotion_candidate")
}

/// Coldest MFU resident (argmin heat_score, older last_used breaks ties).
pub fn coldest_mfu_resident(_mfu_residents: &[ExpertId]) -> Option<ExpertId> {
    todo!("cache::policy::coldest_mfu_resident")
}

/// SWAP predicate: candidate_heat > coldest_heat (strictly greater; tie → evict path).
pub fn should_swap(candidate_heat: u64, coldest_heat: u64) -> bool {
    let _ = (candidate_heat, coldest_heat);
    todo!("cache::policy::should_swap")
}

/// Log-normalized lifetime frequency over CURRENT LRU residents only.
/// C_min/C_max = min/max lifetime_call_count over those residents.
pub fn frequency_score(
    lifetime_call_count: u64,
    min_call_count: u64,
    max_call_count: u64,
) -> f32 {
    let _ = (lifetime_call_count, min_call_count, max_call_count);
    todo!("cache::policy::frequency_score")
}

/// recency_score = 1 / (1 + accesses_since_last_use / 96).
pub fn recency_score(accesses_since_last_use: u64) -> f32 {
    let _ = accesses_since_last_use;
    todo!("cache::policy::recency_score")
}

/// retention_score = 0.25 * frequency_score + 0.75 * recency_score.
pub fn retention_score(frequency_score: f32, recency_score: f32) -> f32 {
    let _ = (frequency_score, recency_score);
    todo!("cache::policy::retention_score")
}

/// LRU resident with the lowest retention_score (eviction victim).
pub fn eviction_victim(_lru_residents: &[ExpertId]) -> Option<ExpertId> {
    todo!("cache::policy::eviction_victim")
}
