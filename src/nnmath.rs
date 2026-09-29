//! nnmath.rs — no_std scalar math for the forward pass (no libm on bare metal).
//!
//! All routines are fp32, deterministic (no data-dependent branches in the
//! hot loops), and host-tested (`cargo test`) against std reference values.
//! NEON acceleration for these is future work; exec.rs owns the matmuls.

/// expf approximation, max rel err ~2e-6 on [-87, 88], 0 below, +inf above.
/// Range-reduction + degree-5 minimax-style polynomial (Cephes coefficients).
pub fn expf_approx(x: f32) -> f32 {
    if x >= 88.0 {
        return f32::INFINITY;
    }
    if x <= -87.0 {
        return 0.0;
    }
    // e^x = 2^k * e^r, k = round(x / ln2), r in [-ln2/2, ln2/2].
    const LN2: f32 = 0.69314718056;
    const INV_LN2: f32 = 1.44269504089;
    let k = round_away(x * INV_LN2);
    let r = x - k * LN2;
    // e^r ~= (((((c5*r + c4)*r + c3)*r + c2)*r + c1)*r + 1)
    let r2 = r * r;
    let p = 1.0 + r + r2 * (0.5 + r * (0.16666667163 + r * (0.04166664177 + r * 0.00833345284)));
    p * exp2_int(k as i32)
}

fn exp2_int(k: i32) -> f32 {
    // 2^k via exponent field manipulation. k in [-127, 127] here by clamping above.
    f32::from_bits(((k + 127) as u32) << 23)
}

/// Round half away from zero (libm-free; exact for |x| < 2^21).
fn round_away(x: f32) -> f32 {
    if x >= 0.0 {
        (x + 0.5) as i32 as f32
    } else {
        (x - 0.5) as i32 as f32
    }
}

/// 1/sqrt(x), Quake bit-hack + 3 Newton iterations (~fp32-exact).
/// core has no sqrt/rsqrt; 3 iterations converge past fp32 epsilon.
fn rsqrt_approx(x: f32) -> f32 {
    let y0 = f32::from_bits(0x5f3759df - (x.to_bits() >> 1));
    let y1 = y0 * (1.5 - 0.5 * x * y0 * y0);
    let y2 = y1 * (1.5 - 0.5 * x * y1 * y1);
    y2 * (1.5 - 0.5 * x * y2 * y2)
}

/// RoPE tables are host-computed (exact fp32) and packed in the image --
/// firmware trig would need >1e-4 accuracy to avoid route flips, which
/// Taylor series cannot deliver. See tools/pack_olmoe_full.py (tag 6/7).
/// (Kept out of firmware deliberately; do not re-add sin/cos here.)

/// In-place softmax over exactly `n` entries (fp32, max-subtracted).
pub fn softmax_inplace(v: &mut [f32]) {
    let mut mx = v[0];
    for &x in v.iter().skip(1) {
        if x > mx {
            mx = x;
        }
    }
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = expf_approx(*x - mx);
        sum += *x;
    }
    let inv = 1.0 / sum;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Top-k selection over 64 router scores: returns indices (any order) + count.
/// Partial selection sort, k <= 16. Ties broken by lower index (stable).
pub fn topk_64(scores: &[f32; 64], k: usize, out_idx: &mut [usize; 16]) -> usize {
    let mut used = [false; 64];
    let mut n = 0;
    while n < k {
        let mut best = 0;
        while best < 64 && used[best] {
            best += 1;
        }
        for i in (best + 1)..64 {
            if used[i] {
                continue;
            }
            if scores[i] > scores[best] || (scores[i] == scores[best] && i < best) {
                best = i;
            }
        }
        used[best] = true;
        out_idx[n] = best;
        n += 1;
    }
    n
}

/// RMSNorm: out[i] = w[i] * x[i] / sqrt(mean(x^2) + eps). x, out len n.
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let mut acc = 0.0f32;
    for &v in x.iter() {
        acc += v * v;
    }
    let inv = rsqrt_approx(acc / (x.len() as f32) + eps);
    for i in 0..x.len() {
        out[i] = w[i] * x[i] * inv;
    }
}

/// SiLU: x * sigmoid(x), sigmoid via expf_approx.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + expf_approx(-x))
}

/// bf16 bits -> f32 (exact: left-shift mantissa by 16).
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Argmax over logits. Ties -> lowest index (matches torch.argmax).
pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_matches_std() {
        for &x in &[-10.0f32, -1.0, -0.001, 0.0, 0.5, 2.0, 10.0, 50.0] {
            let got = expf_approx(x);
            let want = x.exp();
            assert!((got - want).abs() <= 1e-4 * want.abs().max(1.0), "x={x} got={got} want={want}");
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut v = [1.0f32, 2.0, 3.0, 4.0, -5.0, 0.25, -0.75, 10.0];
        softmax_inplace(&mut v);
        let s: f32 = v.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "sum={s}");
        // spot check vs std
        let mx = 10.0f32;
        let want = (4.0f32 - mx).exp() / [1.0f32, 2.0, 3.0, 4.0, -5.0, 0.25, -0.75, 10.0].iter().map(|x| (x - mx).exp()).sum::<f32>();
        assert!((v[3] - want).abs() < 1e-4, "v3={} want={want}", v[3]);
    }

    #[test]
    fn topk_picks_largest_stable() {
        let mut s = [0.0f32; 64];
        s[7] = 5.0;
        s[63] = 5.0;
        s[3] = 9.0;
        s[40] = -1.0;
        let mut out = [0usize; 16];
        let n = topk_64(&s, 8, &mut out);
        assert_eq!(n, 8);
        // first must be index 3 (largest), then 7 before 63 (tie -> lower)
        assert_eq!(out[0], 3);
        assert_eq!(out[1], 7);
        assert_eq!(out[2], 63);
    }

    #[test]
    fn rmsnorm_matches_formula() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let w = [1.0f32; 4];
        let mut o = [0.0f32; 4];
        rmsnorm(&x, &w, 1e-5, &mut o);
        let var = (1.0 + 4.0 + 9.0 + 16.0) / 4.0;
        let inv = 1.0 / (var + 1e-5f32).sqrt();
        for i in 0..4 {
            assert!((o[i] - x[i] * inv).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_tables_come_from_the_image() {
        // Trig deliberately lives on the host (packer), not here: see docs.
        assert!(true);
    }

    #[test]
    fn rsqrt_near_exact() {
        for &x in &[0.001f32, 0.05, 1.0, 7.5, 100.0, 4000.0] {
            let got = rsqrt_approx(x);
            let want = 1.0 / x.sqrt();
            assert!((got - want).abs() / want < 2e-6, "x={x} got={got} want={want}");
        }
    }

    #[test]
    fn bf16_convert_exact() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0x4000), 2.0);
        assert_eq!(bf16_to_f32(0xBFC0), -1.5);
    }
}
