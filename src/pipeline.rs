//! pipeline.rs — orchestrates steps 1→7 in order.
//! 1 token queue → 2 layer counter → 3 gating → 4 dispatch → 5 cache resolve
//! → 6 join on matmul outputs → 7 layer branch (counter.advance():
//! has_more → next layer with new hidden state; last → head/sample/drain
//! → feed back to step 1).

use crate::config::{NUM_LAYERS, TOP_K};
use crate::layer::LayerCounter;

/// Stub entry used by kernel_main until real init lands.
/// Runs the exec self-test (NEON or scalar per build switch) over UART.
/// TODO: replace with run() driving real token/layer/gate/cache/exec/combine/head.
pub fn run_stub() {
    crate::uart::print("pipeline stub: no model loaded\n");
    crate::uart::print("exec backend: ");
    crate::uart::print(crate::exec::backend_name());
    crate::uart::print("\n");
    if crate::exec::selftest() {
        crate::uart::print("exec selftest: PASS\n");
    } else {
        crate::uart::print("exec selftest: FAIL\n");
    }
    let _counter = LayerCounter::new(NUM_LAYERS);
    let _k = TOP_K;
}

/// Real loop shape (stub body): dequeue → per-layer gate/dispatch/resolve/
// exec/combine with `advance()` as the ONLY branch predicate → drain → repeat.
// TODO: implement once storage + exec land.
pub fn run() -> ! {
    todo!("pipeline::run")
}
