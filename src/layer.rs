//! layer.rs — Step 2: bounded layer counter (current layer / total L).
//! This is a bounded INDEX, not a queue. Single branching predicate:
//! `advance()` only (no `is_last` — direction is in the return name).

/// Bounded index over `0..total`. `total` = NUM_LAYERS (OLMoE: 16).
#[derive(Clone, Copy, Debug)]
pub struct LayerCounter {
    current: u32,
    total: u32,
}

impl LayerCounter {
    /// `total` must be >= 1; `current` starts at 0.
    pub const fn new(total: u32) -> Self {
        assert!(total >= 1);
        Self { current: 0, total }
    }

    pub fn current(&self) -> u32 {
        self.current
    }

    pub fn total(&self) -> u32 {
        self.total
    }

    pub fn reset(&mut self) {
        self.current = 0;
    }

    /// Move to the next layer.
    /// Returns `has_more`: true = advanced (more layers remain),
    /// false = was already the last layer (counter unchanged).
    /// Call-site pattern: `if counter.advance() { /* next layer */ } else { /* output head */ }`.
    pub fn advance(&mut self) -> bool {
        if self.current + 1 < self.total {
            self.current += 1;
            true
        } else {
            false
        }
    }
}
