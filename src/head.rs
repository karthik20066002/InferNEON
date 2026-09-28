//! head.rs — Step 7 tail: output head → sample → token drain.
//! Last layer only. Drain feeds the sampled id back to the token queue
//! as the next input (pipeline::run loop).

use crate::token::TokenId;

/// Hidden state → vocab logits. TODO: real head matmul (NEON later).
pub trait OutputHead {
    fn project(&self, hidden_state: &[f32], logits_out: &mut [f32]);
}

/// Logits → token id. TODO: greedy now, top-p/temperature later.
pub trait Sampler {
    fn sample(&self, logits: &[f32]) -> TokenId;
}

/// Emit the token (UART for v1) and hand it back as the next queue input.
pub trait TokenDrain {
    fn emit(&self, token: TokenId) -> TokenId;
}

/// Stub head/sampler/drain bundle.
pub struct StubHead;

impl OutputHead for StubHead {
    fn project(&self, _hidden_state: &[f32], _logits_out: &mut [f32]) {
        todo!("head::StubHead::project")
    }
}

impl Sampler for StubHead {
    fn sample(&self, _logits: &[f32]) -> TokenId {
        todo!("head::StubHead::sample")
    }
}

impl TokenDrain for StubHead {
    fn emit(&self, _token: TokenId) -> TokenId {
        todo!("head::StubHead::emit")
    }
}
