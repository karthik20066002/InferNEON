//! gate.rs — Step 3: gating network, token + hidden state → top-k expert IDs.
//! k is configurable (`config::TOP_K`); static buffers sized by `MAX_K`.

use crate::config::MAX_K;
use crate::token::TokenId;

/// Expert index within one layer (0 .. NUM_EXPERTS_PER_LAYER).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ExpertId(pub u32);

/// Gating boundary. Implementations score all experts then select top-k.
/// TODO: real router matmul + softmax + top-k (keep NEON out of the stub).
pub trait GatingNetwork {
    /// Write exactly `k` ids into `out[..k]` (k ≤ MAX_K). Order is unspecified —
    /// dispatch treats the result as an unordered set.
    fn top_k(&self, token: TokenId, hidden_state: &[f32], k: usize, out: &mut [ExpertId; MAX_K]);
}

/// Placeholder router. TODO: weights + NEON matmul.
pub struct StubGate;

impl GatingNetwork for StubGate {
    fn top_k(&self, _token: TokenId, _hidden_state: &[f32], _k: usize, _out: &mut [ExpertId; MAX_K]) {
        todo!("gate::StubGate::top_k")
    }
}
