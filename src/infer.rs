//! infer.rs — wired generation loop (proof run, not performance).
//!
//! Mirrors transformers OlmoeForCausalLM greedy math exactly, single token at
//! a time, fp32 throughout (matches HF's fp32 accumulation points):
//! embed row -> 16x [rmsnorm -> qkv/o (q,k full-2048 RMSNorms) -> RoPE ->
//! causal SDPA over KV cache -> residual -> rmsnorm -> gate(top8, NO renorm)
//! -> SwiGLU experts x weight -> residual] -> final norm -> head argmax.
//! Prompt is PROMPT_IDS; generates DECODE_STEPS tokens; UART hex IDs.
//! Cache policy intentionally bypassed (fetch-always): this run proves the
//! forward loop, E1 covers policy design in sim. todo!()s elsewhere stay.

use crate::config::*;
use crate::exec;
use crate::fetch;
use crate::nnmath;
use crate::uart;

// ---------------------------------------------------------------------------
// Static state (linker-sized).
// ---------------------------------------------------------------------------

/// KV cache: K and V, [layer, token, head, dim] fp32.
static mut K_CACHE: [f32; 16 * 16 * 16 * 128] = [0.0; 16 * 16 * 16 * 128];
static mut V_CACHE: [f32; 16 * 16 * 16 * 128] = [0.0; 16 * 16 * 16 * 128];

fn k_at(li: usize, t: usize, h: usize, d: usize) -> f32 {
    unsafe { K_CACHE[((li * MAX_TOKENS + t) * NUM_HEADS + h) * HEAD_DIM + d] }
}
fn k_set(li: usize, t: usize, h: usize, d: usize, v: f32) {
    unsafe { K_CACHE[((li * MAX_TOKENS + t) * NUM_HEADS + h) * HEAD_DIM + d] = v; }
}
fn v_at(li: usize, t: usize, h: usize, d: usize) -> f32 {
    unsafe { V_CACHE[((li * MAX_TOKENS + t) * NUM_HEADS + h) * HEAD_DIM + d] }
}
fn v_set(li: usize, t: usize, h: usize, d: usize, v: f32) {
    unsafe { V_CACHE[((li * MAX_TOKENS + t) * NUM_HEADS + h) * HEAD_DIM + d] = v; }
}

// Scratch hidden vectors (reused every layer/token).
static mut H_BUF: [f32; 2048] = [0.0; 2048];
static mut X_BUF: [f32; 2048] = [0.0; 2048];
static mut Q_BUF: [f32; 16 * 128] = [0.0; 16 * 128];
static mut K_BUF: [f32; 16 * 128] = [0.0; 16 * 128];
static mut V_BUF: [f32; 16 * 128] = [0.0; 16 * 128];
static mut A_BUF: [f32; 16 * 128] = [0.0; 16 * 128];
static mut TMP_BUF: [f32; 1024] = [0.0; 1024];
static mut OUT_BUF: [f32; 2048] = [0.0; 2048];
static mut QN_BUF: [f32; 2048] = [0.0; 2048];
static mut KN_BUF: [f32; 2048] = [0.0; 2048];
static mut TMPN_BUF: [f32; 2048] = [0.0; 2048];
static mut SCORE_BUF: [f32; 64] = [0.0; 64];
static mut ATT_SCORE: [f32; 16] = [0.0; 16];

const SCALE: f32 = 0.08838834764; // 128^-0.5
const NORM_EPS: f32 = 1e-5;
const EOS: u32 = 50279;

// ---------------------------------------------------------------------------
// One decoder layer for one token. t = absolute position (KV slot + RoPE).
// Returns via H_BUF (updated in place).
// ---------------------------------------------------------------------------
fn layer_forward<B: crate::storage::BlockStorage>(
    blk: &mut B,
    li: usize,
    t: usize,
    layer_base: u64,
    expert_stride: u64,
) -> Result<(), ()> {
    unsafe {
        // -- attention block --
        fetch::fetch_attn(blk, li as u32)?;
        let attn = &fetch::ATTN_STAGE;
        // split views (bf16 rows): q | k | v | o, each 2048 rows of 2048
        let q_of = 0;
        let k_of = 2048 * 2048;
        let v_of = 2 * 2048 * 2048;
        let o_of = 3 * 2048 * 2048;
        // input norm
        nnmath::rmsnorm(&H_BUF, &fetch::NORMS_F32[li * 2 * 2048..li * 2 * 2048 + 2048], NORM_EPS, &mut X_BUF);
        // q, k, v projections (fused bf16 dots)
        for j in 0..2048 {
            let mut acc = 0.0f32;
            for i in 0..2048 {
                acc += nnmath::bf16_to_f32(attn[q_of + j * 2048 + i]) * X_BUF[i];
            }
            Q_BUF[j] = acc;
        }
        for j in 0..2048 {
            let mut acc = 0.0f32;
            for i in 0..2048 {
                acc += nnmath::bf16_to_f32(attn[k_of + j * 2048 + i]) * X_BUF[i];
            }
            K_BUF[j] = acc;
        }
        for j in 0..2048 {
            let mut acc = 0.0f32;
            for i in 0..2048 {
                acc += nnmath::bf16_to_f32(attn[v_of + j * 2048 + i]) * X_BUF[i];
            }
            V_BUF[j] = acc;
        }
        // q/k norms: per-layer learned RMSNorms (HF order: BEFORE reshape).
        let (qn, kn) = unsafe { (&mut QN_BUF[..], &mut KN_BUF[..]) };
        nnmath::rmsnorm(&Q_BUF, &fetch::QKNORM_F32[li * 2 * 2048..li * 2 * 2048 + 2048], NORM_EPS, qn);
        nnmath::rmsnorm(&K_BUF, &fetch::QKNORM_F32[(li * 2 + 1) * 2048..(li * 2 + 1) * 2048 + 2048], NORM_EPS, kn);
        // RoPE per head (NeoX-style pairs), then KV cache update
        let cos_t = &fetch::ROPE_COS[t * 64..t * 64 + 64];
        let sin_t = &fetch::ROPE_SIN[t * 64..t * 64 + 64];
        for h in 0..16usize {
            for i in 0..64usize {
                let q0 = qn[h * 128 + i];
                let q1 = qn[h * 128 + 64 + i];
                let k0 = kn[h * 128 + i];
                let k1 = kn[h * 128 + 64 + i];
                let c = cos_t[i];
                let s = sin_t[i];
                Q_BUF[h * 128 + i] = q0 * c - q1 * s;
                Q_BUF[h * 128 + 64 + i] = q1 * c + q0 * s;
                let kr0 = k0 * c - k1 * s;
                let kr1 = k1 * c + k0 * s;
                k_set(li, t, h, i, kr0);
                k_set(li, t, h, 64 + i, kr1);
                v_set(li, t, h, i, V_BUF[h * 128 + i]);
                v_set(li, t, h, 64 + i, V_BUF[h * 128 + 64 + i]);
            }
        }
        // causal SDPA over positions 0..=t
        for h in 0..16usize {
            let mut mx = f32::NEG_INFINITY;
            for p in 0..=t {
                let mut s = 0.0f32;
                for d in 0..128usize {
                    s += Q_BUF[h * 128 + d] * k_at(li, p, h, d);
                }
                s *= SCALE;
                ATT_SCORE[p] = s;
                if s > mx {
                    mx = s;
                }
            }
            let mut sum = 0.0f32;
            for p in 0..=t {
                let e = nnmath::expf_approx(ATT_SCORE[p] - mx);
                ATT_SCORE[p] = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for d in 0..128usize {
                let mut acc = 0.0f32;
                for p in 0..=t {
                    acc += ATT_SCORE[p] * inv * v_at(li, p, h, d);
                }
                A_BUF[h * 128 + d] = acc;
            }
        }
        // o_proj + residual
        for j in 0..2048 {
            let mut acc = 0.0f32;
            for i in 0..2048 {
                acc += nnmath::bf16_to_f32(attn[o_of + j * 2048 + i]) * A_BUF[i];
            }
            H_BUF[j] += acc;
        }

        // -- MoE block --
        nnmath::rmsnorm(&H_BUF, &fetch::NORMS_F32[(li * 2 + 1) * 2048..(li * 2 + 1) * 2048 + 2048], NORM_EPS, &mut X_BUF);
        // router: 64 dots over pinned f32 rows, fp32 softmax, top-8, NO renorm
        let rbase = li * 64 * 2048;
        for e in 0..64usize {
            SCORE_BUF[e] = exec::dot(&fetch::ROUTERS_F32[rbase + e * 2048..rbase + e * 2048 + 2048], &X_BUF);
        }
        nnmath::softmax_inplace(&mut SCORE_BUF);
        let mut top: [usize; 16] = [0; 16];
        let nk = nnmath::topk_64(&SCORE_BUF, 8, &mut top);
        for i in 0..2048 {
            OUT_BUF[i] = 0.0;
        }
        for ti in 0..nk {
            let ei = top[ti] as u32;
            let w = SCORE_BUF[top[ti]];
            fetch::fetch_expert(blk, layer_base, expert_stride, ei)?;
            let ex = &fetch::EXPERT_STAGE;
            // gate/up rows [1024,2048], down rows [2048,1024]
            for i in 0..1024usize {
                let mut g = 0.0f32;
                let mut u = 0.0f32;
                for j in 0..2048usize {
                    g += nnmath::bf16_to_f32(ex[i * 2048 + j]) * X_BUF[j];
                    u += nnmath::bf16_to_f32(ex[2 * 1024 * 2048 / 2 + i * 2048 + j]) * X_BUF[j];
                }
                TMP_BUF[i] = nnmath::silu(g) * u;
            }
            for j in 0..2048usize {
                let mut acc = 0.0f32;
                for i in 0..1024usize {
                    acc += nnmath::bf16_to_f32(ex[2 * 2048 * 1024 + j * 1024 + i]) * TMP_BUF[i];
                }
                OUT_BUF[j] += w * acc;
            }
        }
        for j in 0..2048 {
            H_BUF[j] += OUT_BUF[j];
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Full forward for one token id at position t. Returns nothing; H_BUF holds
// the final hidden state (post final-norm). KV appended.
// ---------------------------------------------------------------------------
fn forward_token<B: crate::storage::BlockStorage>(
    blk: &mut B,
    tok: u32,
    t: usize,
    layer_bases: &[u64; 16],
    expert_stride: u64,
) -> Result<(), ()> {
    unsafe {
        fetch::fetch_embed_row(blk, tok, &mut H_BUF)?;
        for li in 0..16usize {
            layer_forward(blk, li, t, layer_bases[li], expert_stride)?;
        }
        // final norm
        let tmp = unsafe { &mut TMPN_BUF[..] };
        nnmath::rmsnorm(&H_BUF, &fetch::NORMS_F32[32 * 2048..33 * 2048], NORM_EPS, tmp);
        H_BUF.copy_from_slice(tmp);
    }
    Ok(())
}

/// Greedy head: stream 8 vocab shards, running argmax. Returns token id.
fn greedy_head<B: crate::storage::BlockStorage>(blk: &mut B) -> Result<u32, ()> {
    let mut best_id = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    unsafe {
        for s in 0..fetch::HEAD_SHARDS {
            fetch::fetch_head_shard(blk, s)?;
            let shard = &fetch::HEAD_STAGE;
            for r in 0..fetch::HEAD_SHARD_ROWS {
                let mut acc = 0.0f32;
                for i in 0..2048usize {
                    acc += nnmath::bf16_to_f32(shard[r * 2048 + i]) * H_BUF[i];
                }
                if acc > best_val {
                    best_val = acc;
                    best_id = (s * fetch::HEAD_SHARD_ROWS + r) as u32;
                }
            }
        }
    }
    Ok(best_id)
}

/// Wired run: prefill PROMPT_IDS, decode DECODE_STEPS greedy. UART hex ids.
/// Returns generated ids packed into out (up to DECODE_STEPS).
pub fn run<B: crate::storage::BlockStorage>(
    blk: &mut B,
    layer_bases: &[u64; 16],
    expert_stride: u64,
    out: &mut [u32],
) -> Result<usize, ()> {
    let mut n_gen = 0usize;
    // prefill prompt (KV fill, no output)
    for (t, tok) in PROMPT_IDS.iter().enumerate() {
        forward_token(blk, *tok, t, layer_bases, expert_stride)?;
    }
    let mut t = PROMPT_IDS.len();
    let mut tok = 0u32;
    // first decode id comes from the last prefill hidden state: recompute
    // head over it (forward_token left H_BUF at last prompt token).
    for _ in 0..DECODE_STEPS {
        let id = greedy_head(blk)?;
        uart::print("tok: ");
        uart::print_hex_u32(id);
        uart::print("\n");
        if n_gen < out.len() {
            out[n_gen] = id;
        }
        n_gen += 1;
        if id == EOS {
            break;
        }
        tok = id;
        forward_token(blk, tok, t, layer_bases, expert_stride)?;
        t += 1;
        if t >= MAX_TOKENS {
            break;
        }
    }
    Ok(n_gen)
}
