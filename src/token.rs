//! token.rs — Step 1: FIFO token queue, dequeues the next token.
//!
//! Autoregressive feedback: `head::TokenDrain` feeds the sampled token id
//! back here as `token n+1 input` (RAW hazard noted in the arch diagram —
//! the next iteration must not dequeue before the drain enqueues).

/// Opaque token id (vocab index).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TokenId(pub u32);

/// FIFO queue boundary. Backing for v1 is a fixed ring (no alloc).
pub trait TokenQueue {
    fn enqueue(&mut self, token: TokenId) -> bool;
    fn dequeue(&mut self) -> Option<TokenId>;
    fn is_empty(&self) -> bool;
    fn len(&self) -> usize;
}

/// Fixed-capacity ring buffer stub. TODO: head/tail/wraparound + full/empty disambiguation.
pub struct ArrayTokenQueue<const N: usize> {
    buf: [Option<TokenId>; N],
    head: usize,
    tail: usize,
    count: usize,
}

impl<const N: usize> ArrayTokenQueue<N> {
    pub const fn new() -> Self {
        Self {
            buf: [None; N],
            head: 0,
            tail: 0,
            count: 0,
        }
    }
}

impl<const N: usize> TokenQueue for ArrayTokenQueue<N> {
    fn enqueue(&mut self, token: TokenId) -> bool {
        if self.count >= N {
            return false;
        }
        self.buf[self.tail] = Some(token);
        self.tail = (self.tail + 1) % N;
        self.count += 1;
        true
    }
    fn dequeue(&mut self) -> Option<TokenId> {
        if self.count == 0 {
            return None;
        }
        let t = self.buf[self.head].take();
        self.head = (self.head + 1) % N;
        self.count -= 1;
        t
    }
    fn is_empty(&self) -> bool {
        self.count == 0
    }
    fn len(&self) -> usize {
        self.count
    }
}
