//! combine.rs — Step 6: join/combine barrier.
//! Waits for all k experts' MATMUL OUTPUTS, not just cache residency.
//! A "resident but not yet computed" expert must NOT release the barrier.

use crate::config::MAX_K;
use crate::exec::MatmulOutput;

/// Weighted-sum reduction of the k matmul outputs into the next hidden state.
/// v1 single-core: the running firmware inlines this accumulation directly
/// in infer.rs (weights × expert outputs + residual). This symbol becomes
/// the real barrier join when N-core lands; until then it must not be
/// called (nothing calls it — verified by grep at wiring time).
pub fn join_combine(_outputs: &[MatmulOutput], _k: usize, _hidden_out: &mut [f32]) {
    todo!("combine::join_combine — N-core barrier join; v1 accumulates inline in infer.rs")
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

    /// Single core: arrival == release; returns immediately. Becomes a real
    /// blocking barrier under N-core.
    pub fn wait_for_outputs(&self, _arrived: usize) {
        let _ = self.expected;
    }
}
