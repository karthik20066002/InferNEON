//! exec.rs — expert execution: NEON matmul kernels + scalar reference.
//!
//! Build switch: Cargo feature `neon` (default ON).
//!   `just build` → NEON on aarch64, scalar elsewhere.
//!   `just build 0` (→ `--no-default-features`) → scalar everywhere.
//! The NEON path compiles only for `target_arch = "aarch64"`; host builds
//! (`cargo test`) always exercise the scalar reference.
//!
//! Kernels here are the SwiGLU FFN pieces (OLMoE expert shapes): dot, GEMV
//! (row-major), SiLU, full expert forward. Weight residency (cache/storage
//! wiring) is still TODO — `StubExecutor::execute` stays a stub until the
//! fetch path lands. What runs TODAY: free functions + boot `selftest()`.

use crate::gate::ExpertId;

/// Opaque per-expert matmul output handle. Combine waits on THESE, not on
/// cache residency (see combine.rs).
#[derive(Clone, Copy, Debug)]
pub struct MatmulOutput {
    pub expert: ExpertId,
    /// TODO: pointer/len into scratch output buffer (decide ownership later).
    pub output_slot: u32,
}

/// Execute one resident expert's FFN over the current hidden state.
/// TODO: wire resident weights (cache/storage) into expert_forward below.
pub trait ExpertExecute {
    fn execute(&self, expert: ExpertId, hidden_state: &[f32]) -> MatmulOutput;
}

/// Stub executor — always panics. Keeps link shape until weight fetch lands.
pub struct StubExecutor;

impl ExpertExecute for StubExecutor {
    fn execute(&self, _expert: ExpertId, _hidden_state: &[f32]) -> MatmulOutput {
        todo!("exec::StubExecutor::execute — weight fetch not wired yet (kernels themselves run; see selftest)")
    }
}

/// Active backend label for the boot log.
pub fn backend_name() -> &'static str {
    if cfg!(all(target_arch = "aarch64", feature = "neon")) {
        "neon"
    } else {
        "scalar"
    }
}

// ---------------------------------------------------------------------------
// Scalar reference (always compiled; IS the backend when neon is off / off-target).
// ---------------------------------------------------------------------------

/// Naive dot product. Reference for the NEON kernel.
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Row-major GEMV: y[r] = dot(matrix[r*cols..(r+1)*cols], x).
pub fn gemv_scalar(matrix: &[f32], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    assert_eq!(matrix.len(), rows * cols);
    assert_eq!(x.len(), cols);
    assert_eq!(y.len(), rows);
    for r in 0..rows {
        y[r] = dot_scalar(&matrix[r * cols..(r + 1) * cols], x);
    }
}

/// SiLU(x) = x * sigmoid(x). Uses libm (no_std-compatible) for exp.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + libm::expf(-x))
}

/// Full SwiGLU expert forward (scalar): gate/up are inter×hidden row-major,
/// down is hidden×inter row-major. tmp is inter-dim scratch.
pub fn expert_forward_scalar(
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    hidden_dim: usize,
    inter_dim: usize,
    x: &[f32],
    tmp: &mut [f32],
    out: &mut [f32],
) {
    assert_eq!(x.len(), hidden_dim);
    assert_eq!(tmp.len(), inter_dim);
    assert_eq!(out.len(), hidden_dim);
    assert_eq!(gate_w.len(), inter_dim * hidden_dim);
    assert_eq!(up_w.len(), inter_dim * hidden_dim);
    assert_eq!(down_w.len(), hidden_dim * inter_dim);
    for i in 0..inter_dim {
        let g = dot_scalar(&gate_w[i * hidden_dim..(i + 1) * hidden_dim], x);
        let u = dot_scalar(&up_w[i * hidden_dim..(i + 1) * hidden_dim], x);
        tmp[i] = silu(g) * u;
    }
    gemv_scalar(down_w, hidden_dim, inter_dim, tmp, out);
}

// ---------------------------------------------------------------------------
// NEON path (aarch64 + feature only).
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
use core::arch::aarch64::*;

/// Vectorized dot: 4-lane FMA accumulation + scalar tail.
/// SAFETY: pure SIMD over checked slices; caller guarantees len equality
/// (asserted in the safe wrapper).
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut i = 0;
    let mut acc = vdupq_n_f32(0.0);
    while i + 4 <= n {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        acc = vfmaq_f32(acc, va, vb);
        i += 4;
    }
    let mut sum = vaddvq_f32(acc);
    while i < n {
        sum += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
    sum
}

/// Row-major GEMV over NEON dots.
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn gemv_neon(matrix: &[f32], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    assert_eq!(matrix.len(), rows * cols);
    assert_eq!(x.len(), cols);
    assert_eq!(y.len(), rows);
    for r in 0..rows {
        // SAFETY: row slice and x share length cols by the asserts above.
        y[r] = unsafe { dot_neon(&matrix[r * cols..(r + 1) * cols], x) };
    }
}

// ---------------------------------------------------------------------------
// Public entry points: NEON where available+enabled, scalar otherwise.
// ---------------------------------------------------------------------------

/// Dot product (NEON on aarch64+neon, scalar otherwise).
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        // SAFETY: lengths asserted equal above.
        unsafe { dot_neon(a, b) }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
    {
        dot_scalar(a, b)
    }
}

/// Row-major GEMV (NEON on aarch64+neon, scalar otherwise).
pub fn gemv(matrix: &[f32], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        gemv_neon(matrix, rows, cols, x, y);
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
    {
        gemv_scalar(matrix, rows, cols, x, y);
    }
}

/// Full SwiGLU expert forward (vectorized GEMVs; scalar SiLU — NEON exp is TODO).
pub fn expert_forward(
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    hidden_dim: usize,
    inter_dim: usize,
    x: &[f32],
    tmp: &mut [f32],
    out: &mut [f32],
) {
    assert_eq!(x.len(), hidden_dim);
    assert_eq!(tmp.len(), inter_dim);
    assert_eq!(out.len(), hidden_dim);
    for i in 0..inter_dim {
        let g = dot(&gate_w[i * hidden_dim..(i + 1) * hidden_dim], x);
        let u = dot(&up_w[i * hidden_dim..(i + 1) * hidden_dim], x);
        tmp[i] = silu(g) * u;
    }
    gemv(down_w, hidden_dim, inter_dim, tmp, out);
}

// ---------------------------------------------------------------------------
// Boot self-test (no_std-safe): exact small constants + neon-vs-scalar cross-check.
// ---------------------------------------------------------------------------

/// Runs at boot from pipeline::run_stub. True = kernels healthy.
pub fn selftest() -> bool {
    // 8-wide exact: 1*8+2*7+...+8*1 = 120.
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    if (dot(&a, &b) - 120.0).abs() > 1e-4 {
        return false;
    }
    // Tail path (len 6, not multiple of 4): 6+10+12+12+10+6 = 56.
    let a6 = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b6 = [6.0f32, 5.0, 4.0, 3.0, 2.0, 1.0];
    if (dot(&a6, &b6) - 56.0).abs() > 1e-4 {
        return false;
    }
    // GEMV: [[1,2,3,4],[5,6,7,8]] x [1,1,1,1] = [10,26].
    let m = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = [1.0f32; 4];
    let mut y = [0.0f32; 2];
    gemv(&m, 2, 4, &x, &mut y);
    if (y[0] - 10.0).abs() > 1e-4 || (y[1] - 26.0).abs() > 1e-4 {
        return false;
    }
    // SiLU sanity: silu(0) == 0 exactly.
    if silu(0.0) != 0.0 {
        return false;
    }
    // On NEON targets, cross-check vector vs scalar on the tail case.
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        let neon = unsafe { dot_neon(&a6, &b6) };
        let scal = dot_scalar(&a6, &b6);
        if (neon - scal).abs() > 1e-4 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_exact_constants() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        assert!((dot(&a, &b) - 120.0).abs() < 1e-4);
        let a6 = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b6 = [6.0f32, 5.0, 4.0, 3.0, 2.0, 1.0];
        assert!((dot(&a6, &b6) - 56.0).abs() < 1e-4);
    }

    #[test]
    fn gemv_known_answer() {
        let m = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x = [1.0f32; 4];
        let mut y = [0.0f32; 2];
        gemv(&m, 2, 4, &x, &mut y);
        assert!((y[0] - 10.0).abs() < 1e-4 && (y[1] - 26.0).abs() < 1e-4);
    }

    #[test]
    fn expert_forward_identity_weights() {
        // hidden=2, inter=2, all weights identity: out[i] = silu(x[i]) * x[i].
        let ident = [1.0f32, 0.0, 0.0, 1.0];
        let x = [1.0f32, 2.0];
        let mut tmp = [0.0f32; 2];
        let mut out = [0.0f32; 2];
        expert_forward(&ident, &ident, &ident, 2, 2, &x, &mut tmp, &mut out);
        let e0 = 1.0f32 / (1.0 + (-1.0f32).exp()) * 1.0;
        let e1 = 2.0f32 / (1.0 + (-2.0f32).exp()) * 2.0;
        assert!((out[0] - e0).abs() < 1e-5, "out0={} e0={}", out[0], e0);
        assert!((out[1] - e1).abs() < 1e-5, "out1={} e1={}", out[1], e1);
    }

    #[test]
    fn selftest_passes_on_host() {
        assert!(selftest());
    }
}
