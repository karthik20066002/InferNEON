//! combine.rs — Step 6: join/combine barrier.
//! Waits for all k experts' MATMUL OUTPUTS, not just cache residency.
//! A "resident but not yet computed" expert must NOT release the barrier.

use crate::config::MAX_K;
use crate::exec::MatmulOutput;

/// Weighted-sum reduction of the k matmul outputs into the next hidden state.
/// TODO: gating weights + accumulate (NEON later); single-core sums in place.
pub fn join_combine(_outputs: &[MatmulOutput], _k: usize, _hidden_out: &mut [f32]) {
    todo!("combine::join_combine — barrier on matmul outputs")
}

/// Barrier state stub for the join (reusable across layers via generation).
/// TODO: back with sync::Barrier once multicore lands.
pub struct JoinBarrier {
    expected: usize,
}

impl JoinBarrier {
    pub const fn new(expected: usize) -> Self {
        assert!(expected <= MAX_K);
        Self { expected }
    }

    /// TODO: block until `expected` outputs have arrived, then release.
    pub fn wait_for_outputs(&self, _arrived: usize) {
        todo!("combine::JoinBarrier::wait_for_outputs")
    }
}
