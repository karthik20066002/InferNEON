"""Layer-streaming trace capturer for Qwen3.6 GPTQ-INT4.

Why streaming: the box (14 GB RAM / 8 GB VRAM) cannot stage the 22.7 GB
checkpoint through transformers' loader (OOM x3, journal-confirmed). So we
never hold the whole model: embed/norm/lm_head/rotary/routers/gates/norms
stay resident (~3 GB), and exactly ONE decoder layer's weights are
materialized at a time from the local safetensors shards via safe_open
slices + the proven INT4 dequant (capture/dequant.py).

What stays HF's (exact numerics, no reimplementation): every submodule
forward (GatedDeltaNet, GQA attention, RMSNorm, TopKRouter, SwiGLU MLP),
DynamicCache, rotary embeddings, greedy sampling.
What is ours: residency (which layer is material), hook recording, files.

Decode is token-by-token for prefill AND decode, so no attention masks are
ever needed (single query over cache is causal by construction; no sliding
window in this config). Reference: Qwen3_5MoeTextModel.forward calling
convention (position_ids [4,1,1], rotary once per step, shared DynamicCache).

Recorded per decode token per layer 0-39: h_in, h_postmix (fp16),
top-8 ids in router order, 8 gate scores, full 256 router logits (fp16).
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    ROPE_INIT_FUNCTIONS,
)

from capture.dequant import dequant_linear
from capture.weight_cache import WeightCache

PREFIX = "model.language_model."

# RAM memo cap for dequantized tensors (mixers + experts). Pure-function
# cache: same checkpoint bytes always yield identical fp16, so hits cannot
# change numerics; eviction only costs recompute. Callers MUST NOT mutate
# returned tensors (all current uses copy onward: .to(dev), .T, copy_).
MEMO_MAX_BYTES = 1536 * 1024**2


class StreamCapture:
    def __init__(self, model_dir: str, device: str = "cuda", dtype=torch.float16,
                 weight_cache_bytes: int = 0):
        self.dir = model_dir
        self.device = torch.device(device)
        self.dtype = dtype
        self.weight_cache = WeightCache(weight_cache_bytes)
        self._memo: OrderedDict = OrderedDict()
        self._memo_bytes = 0
        full = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        self.cfg = full.text_config if hasattr(full, "text_config") else full
        self.nlayers = self.cfg.num_hidden_layers
        self.hidden = self.cfg.hidden_size
        with torch.device("meta"):
            self.model = Qwen3_5MoeForCausalLM(self.cfg)
        self.model.eval()
        self.index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
        self.wmap = self.index["weight_map"]
        self.handles = {}
        for fn in sorted({v for v in self.wmap.values()}):
            if fn.endswith(".safetensors"):
                self.handles[fn] = safe_open(os.path.join(model_dir, fn), framework="pt")
        self.layer_types = self.cfg.layer_types
        self._materialize_resident()

    # -- weight access ----------------------------------------------------
    # KEY CONVENTION (asserted in tests): `_get` takes a FULL key
    # ("...gate_proj.weight"); `_dequant_or_dense` takes a BASE key
    # ("...gate_proj") and appends ".weight" for the dense fallback itself.
    def _get(self, ckpt_name: str) -> torch.Tensor:
        fn = self.wmap[ckpt_name]
        return self.handles[fn].get_slice(ckpt_name)[:]

    def _dequant_or_dense(self, base: str) -> torch.Tensor:
        """Dequantized fp16 [in, out] for a linear, dense passthrough if stored BF16.

        Results are memoized in RAM (FIFO, MEMO_MAX_BYTES): the same base key
        always maps to identical bytes, so a hit is bit-identical to a fresh
        dequant. This removes ~960 float64 CPU dequants per decode token
        after warmup. Returned tensors are read-only to callers.
        """
        hit = self._memo.get(base)
        if hit is not None:
            self._memo.move_to_end(base)
            return hit
        try:
            qw = self._get(base + ".qweight")
        except Exception:
            w = self._get(base + ".weight").to(self.dtype)
        else:
            qz = self._get(base + ".qzeros")
            sc = self._get(base + ".scales")
            gi = self._get(base + ".g_idx")
            w = dequant_linear(qw, qz, sc, gi, self.dtype)
        self._memo[base] = w
        self._memo_bytes += w.numel() * w.element_size()
        while self._memo_bytes > MEMO_MAX_BYTES and len(self._memo) > 1:
            _, old = self._memo.popitem(last=False)
            self._memo_bytes -= old.numel() * old.element_size()
        return w

    def _materialize_resident(self):
        """embed, final norm, lm_head, rotary, all routers/gates/norms. Small."""
        m = self.model.model
        dev, dt = self.device, self.dtype
        with torch.no_grad():
            # keys under model.language_model.*; lm_head sits at the top level.
            m.embed_tokens.weight = torch.nn.Parameter(
                self._get(f"{PREFIX}embed_tokens.weight").to(dev).to(dt))
            m.norm.weight = torch.nn.Parameter(
                self._get(f"{PREFIX}norm.weight").to(dev).to(dt))
            lm = self._dequant_or_dense("lm_head")
            self.model.lm_head.weight = torch.nn.Parameter(lm.to(dev).to(dt))
            for li in range(self.nlayers):
                p = f"{PREFIX}layers.{li}."
                lay = m.layers[li]
                lay.input_layernorm.weight = torch.nn.Parameter(
                    self._get(p + "input_layernorm.weight").to(dev).to(dt))
                lay.post_attention_layernorm.weight = torch.nn.Parameter(
                    self._get(p + "post_attention_layernorm.weight").to(dev).to(dt))
                lay.mlp.gate.weight = torch.nn.Parameter(
                    self._get(p + "mlp.gate.weight").to(dev).to(dt))
                lay.mlp.shared_expert_gate.weight = torch.nn.Parameter(
                    self._get(p + "mlp.shared_expert_gate.weight").to(dev).to(dt))
            # fused expert staging buffers on CUDA (reused every step)
            ne, ni, nh = self.cfg.num_experts, self.cfg.moe_intermediate_size, self.hidden
            m._fused_gate_up = torch.zeros(ne, 2 * ni, nh, device=dev, dtype=dt)
            m._fused_down = torch.zeros(ne, nh, ni, device=dev, dtype=dt)
            self._materialize_rotary(m, dev)
        self._attach_fused(m)

    def _materialize_rotary(self, m, dev):
        """Config-derived rotary tables onto CUDA (not checkpoint weights).

        Rotary inv_freq/original_inv_freq are nn.Buffers (persistent=False),
        not Parameters: the meta-device init leaves them on 'meta' and the
        per-layer loader (named_parameters() only) never touches them.
        Qwen3.5's matmul-based mrope does NOT fail loudly on meta input:
        meta@real silently yields zeros, i.e. cos=1/sin=0, so every
        full-attention layer ran position-blind; under deterministic
        algorithms the same path yields NaN. Buffers with persistent=False
        are absent from the checkpoint, so config-derived values are exactly
        what a real HF init produces (verified: no inv_freq key for rotary
        in model.safetensors.index.json weight_map).
        """
        rope = m.rotary_emb
        if getattr(rope, "inv_freq", None) is None or not rope.inv_freq.is_meta:
            return
        if rope.rope_type == "default":
            inv_freq, _ = type(rope).compute_default_rope_parameters(self.cfg)
        else:
            inv_freq, _ = ROPE_INIT_FUNCTIONS[rope.rope_type](self.cfg)
        rope.inv_freq = torch.nn.Buffer(inv_freq.to(dev), persistent=False)
        rope.original_inv_freq = torch.nn.Buffer(inv_freq.clone().to(dev), persistent=False)

    def _attach_fused(self, m):
        for li in range(self.nlayers):
            ex = m.layers[li].mlp.experts
            ex.gate_up_proj = torch.nn.Parameter(m._fused_gate_up, requires_grad=False)
            ex.down_proj = torch.nn.Parameter(m._fused_down, requires_grad=False)

    # -- per-layer transient weights --------------------------------------
    def _load_mixer_and_shared(self, li: int):
        """Materialize layer li's mixer linears + shared-expert MLP."""
        m = self.model.model
        lay = m.layers[li]
        mix = lay.linear_attn if self.layer_types[li] == "linear_attention" else lay.self_attn
        dev, dt = self.device, self.dtype
        p = f"{PREFIX}layers.{li}."
        with torch.no_grad():
            self._load_mixer(mix, li)
            se = lay.mlp.shared_expert
            for proj in ("gate_proj", "up_proj", "down_proj"):
                mod = getattr(se, proj)
                w = self._dequant_or_dense(p + f"mlp.shared_expert.{proj}")
                if tuple(w.shape) == (mod.in_features, mod.out_features):
                    w = w.T.contiguous()
                elif tuple(w.shape) != (mod.out_features, mod.in_features):
                    raise AssertionError(
                        f"shared_expert.{proj}: {tuple(w.shape)} matches neither "
                        f"(in,out)={(mod.in_features, mod.out_features)} nor "
                        f"(out,in)={(mod.out_features, mod.in_features)}")
                mod.weight = torch.nn.Parameter(w.to(dev).to(dt))

    def _fill_experts(self, li: int, need_experts, *, use_cache=True):
        """Fill fused gate_up/down slices for the routed hit set only (peak = 1 layer)."""
        m = self.model.model
        dev, dt = self.device, self.dtype
        p = f"{PREFIX}layers.{li}."
        ni, nh = self.cfg.moe_intermediate_size, self.cfg.hidden_size
        with torch.no_grad():
            for ei in need_experts:
                ep = p + f"mlp.experts.{int(ei)}."
                def load():
                    return tuple(self._dequant_or_dense(ep + proj).to(dev)
                                 for proj in ('gate_proj', 'up_proj', 'down_proj'))
                if use_cache:
                    g, u, d = self.weight_cache.get((li, int(ei)), load)
                else:
                    g, u, d = load()
                # experts are stored [in, out]; fused buffers are [out, in]
                assert tuple(g.shape) == (nh, ni), (ei, 'gate', tuple(g.shape))
                assert tuple(u.shape) == (nh, ni), (ei, 'up', tuple(u.shape))
                assert tuple(d.shape) == (ni, nh), (ei, 'down', tuple(d.shape))
                m._fused_gate_up[int(ei), :ni].copy_(g.T.to(dev).to(dt))
                m._fused_gate_up[int(ei), ni:].copy_(u.T.to(dev).to(dt))
                m._fused_down[int(ei)].copy_(d.T.to(dev).to(dt))

    QUANT_LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv",
                     "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")

    def _load_mixer(self, mix, li: int):
        """Assign weights to every meta parameter under the mixer.

        Covers Linear projections (dequantized) AND non-Linear params
        (GatedDeltaNet: A_log, dt_bias, conv1d.weight, norm.weight; attention:
        q_norm/k_norm). `_dequant_or_dense` takes a BASE key (no .weight);
        `_get` takes the full key.
        """
        dev, dt = self.device, self.dtype
        kind = "linear_attn" if self.layer_types[li] == "linear_attention" else "self_attn"
        with torch.no_grad():
            for rel, param in mix.named_parameters():
                if not param.is_meta:
                    continue
                base = f"{PREFIX}layers.{li}.{kind}.{rel}"
                stem = base[: -len(".weight")] if base.endswith(".weight") else base
                mod = mix.get_submodule(rel.rsplit(".", 1)[0]) if "." in rel else mix
                if isinstance(mod, torch.nn.Linear):
                    want_in, want_out = mod.in_features, mod.out_features
                elif param.ndim == 2:
                    want_in, want_out = param.shape[1], param.shape[0]
                else:
                    # 0-d (A_log, dt_bias) or 3-d (conv1d): identity orientation
                    w = self._get(base)
                    if tuple(w.shape) != tuple(param.shape):
                        raise AssertionError(f"{base}: {tuple(w.shape)} != {tuple(param.shape)}")
                    owner = mix.get_submodule(rel.rsplit(".", 1)[0]) if "." in rel else mix
                    setattr(owner, rel.rsplit(".", 1)[-1],
                            torch.nn.Parameter(w.to(dev).to(dt), requires_grad=False))
                    continue
                w = self._dequant_or_dense(stem)  # dense [out, in] or dequant [?, ?]
                if tuple(w.shape) == (want_in, want_out):
                    w = w.T.contiguous()          # dequant came back [in, out]
                elif tuple(w.shape) != (want_out, want_in):
                    raise AssertionError(
                        f"{base}: weight {tuple(w.shape)} matches neither "
                        f"(in,out)={(want_in, want_out)} nor (out,in)={(want_out, want_in)}")
                owner = mix.get_submodule(rel.rsplit(".", 1)[0]) if "." in rel else mix
                setattr(owner, rel.rsplit(".", 1)[-1],
                        torch.nn.Parameter(w.to(dev).to(dt), requires_grad=False))

    def _mixer_key(self, li: int, relname: str) -> str:
        kind = "linear_attn" if self.layer_types[li] == "linear_attention" else "self_attn"
        return f"{PREFIX}layers.{li}.{kind}.{relname}"  # base; caller appends .weight

    def _drop_layer(self, li: int):
        """Return layer li to meta (free the transient). Resident parts untouched."""
        m = self.model.model
        lay = m.layers[li]
        mix = lay.linear_attn if self.layer_types[li] == "linear_attention" else lay.self_attn
        with torch.no_grad():
            for rel, param in mix.named_parameters():
                if param.is_meta:
                    continue
                owner = mix.get_submodule(rel.rsplit(".", 1)[0]) if "." in rel else mix
                setattr(owner, rel.rsplit(".", 1)[-1],
                        torch.nn.Parameter(torch.empty(param.shape, device="meta",
                                                      dtype=self.dtype), requires_grad=False))
            se = lay.mlp.shared_expert
            for proj in ("gate_proj", "up_proj", "down_proj"):
                mod = getattr(se, proj)
                mod.weight = torch.nn.Parameter(
                    torch.empty(mod.weight.shape, device="meta", dtype=self.dtype))
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # -- batched prompt prefill ---------------------------------------------
    def prefill(self, token_ids, recs):
        """Process a whole prompt layer-major with one streamed layer at a time.

        `recs` has one record per prompt token. This preserves per-token routing
        traces while avoiding one complete 40-layer stream per prompt token.
        The resulting DynamicCache is ready for token-at-a-time decode via
        `step()`.
        """
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_causal_mask

        m = self.model.model
        dev = self.device
        # Each prompt owns fresh attention/DeltaNet state. Weight tensors alone
        # may persist between prompts; never reuse KV/recurrent state here.
        self.cache = DynamicCache(config=self.cfg)
        ids = torch.tensor([list(token_ids)], device=dev)
        h = m.embed_tokens(ids)
        seq_len = h.shape[1]
        text_pos = torch.arange(seq_len, device=dev).view(1, seq_len)
        rot = m.rotary_emb(h, text_pos.expand(3, 1, seq_len))

        with torch.no_grad():
            for li in range(self.nlayers):
                lay = m.layers[li]
                for ti in range(seq_len):
                    recs[ti]["h_in"].append(h[0, ti].to(torch.float16).cpu())

                self._load_mixer_and_shared(li)
                residual_in = h
                x = lay.input_layernorm(h)
                if self.layer_types[li] == "linear_attention":
                    x = lay.linear_attn(
                        hidden_states=x,
                        cache_params=self.cache,
                        attention_mask=None,
                    )
                else:
                    # Full-attention layers need the causal prompt mask;
                    # omitting it would make prefill bidirectional.
                    mask = create_causal_mask(
                        config=self.cfg,
                        inputs_embeds=x,
                        attention_mask=None,
                        past_key_values=self.cache,
                        position_ids=text_pos,
                        layer_idx=lay.self_attn.layer_idx,
                        allow_is_causal_skip=False,
                    )
                    x, _ = lay.self_attn(
                        hidden_states=x,
                        attention_mask=mask,
                        position_ids=text_pos,
                        past_key_values=self.cache,
                        position_embeddings=rot,
                    )
                h_after_mix = residual_in + x

                router_input = lay.post_attention_layernorm(h_after_mix)
                for ti in range(seq_len):
                    recs[ti]["h_postmix"].append(
                        h_after_mix[0, ti].to(torch.float16).cpu()
                    )
                    recs[ti]["router_input"].append(
                        router_input[0, ti].to(torch.float16).cpu()
                    )

                logits, scores, idx = lay.mlp.gate(
                    router_input.reshape(-1, self.hidden)
                )
                for ti in range(seq_len):
                    recs[ti]["logits256"].append(logits[ti].to(torch.float16).cpu())
                    recs[ti]["top8"].append(idx[ti].cpu())
                    recs[ti]["scores8"].append(scores[ti].to(torch.float32).cpu())

                # Do not fill the bounded decode cache with the much larger
                # union of prefill experts. It remains warm across prompts.
                self._fill_experts(li, idx.reshape(-1).unique().tolist(), use_cache=False)
                out = lay.mlp(router_input)
                if isinstance(out, tuple):
                    out = out[0]
                h = h_after_mix + out
                self._drop_layer(li)

            h = m.norm(h)
            logits = self.model.lm_head(h[:, -1:, :])
        return self.cache, logits[0, 0].float().cpu()

    # -- one decode step over the whole stack -------------------------------
    def step(self, token_id: int, rec: dict | None, pos: int):
        """Run ONE token through all 40 layers (HF forward math, streamed weights).

        rec (optional) is filled in place: h_in[L], h_postmix[L], top8[L],
        scores8[L], logits256[L] — all on CPU, fp16/int64.
        """
        from transformers.cache_utils import DynamicCache
        m = self.model.model
        dev, dt = self.device, self.dtype
        if not hasattr(self, "cache"):
            self.cache = DynamicCache(config=self.cfg)
        with torch.no_grad():
            ids = torch.tensor([[token_id]], device=dev)
            h = m.embed_tokens(ids)
            # text rotary takes 3 rows (T,H,W), recomposes internally; the
            # attention layers get the T row (shape [bs, 1]).
            text_pos = torch.tensor([[pos]], device=dev)
            rot = m.rotary_emb(h, text_pos.expand(3, 1, 1))
            for li in range(self.nlayers):
                lay = m.layers[li]
                if rec is not None:
                    rec["h_in"].append(h[0, 0].to(torch.float16).cpu())
                self._load_mixer_and_shared(li)
                attn_mask = None  # single query over cache: no mask needed
                residual_in = h
                x = lay.input_layernorm(h)
                if self.layer_types[li] == "linear_attention":
                    x = lay.linear_attn(hidden_states=x, cache_params=self.cache,
                                         attention_mask=attn_mask)
                else:
                    x, _ = lay.self_attn(hidden_states=x, attention_mask=attn_mask,
                                         position_ids=text_pos,
                                         past_key_values=self.cache,
                                         position_embeddings=rot)
                h_after_mixer = residual_in + x
                h = h_after_mixer  # residual + mixer output, set by inline block
                h = self._step_mlp_residual(lay, h, li, rec)
                self._drop_layer(li)
            h = m.norm(h)
            logits = self.model.lm_head(h[:, -1:, :])
        return self.cache, logits[0, 0].float().cpu()

    def _step_mlp_residual(self, lay, h_after_mix, li, rec):
        # mirrors Qwen3_5MoeDecoderLayer.forward exactly
        residual = h_after_mix
        # h_postmix is the token-mixer output plus its residual, immediately
        # before the MoE pre-router norm. Keep the normalized router input
        # separately so validation can reproduce the exact live operation.
        if rec is not None:
            rec["h_postmix"].append(h_after_mix[0, 0].to(torch.float16).cpu())
        x = lay.post_attention_layernorm(h_after_mix)
        if rec is not None:
            rec["router_input"].append(x[0, 0].to(torch.float16).cpu())
        # router (resident) → top-8, scores
        logits, scores, idx = lay.mlp.gate(x.view(-1, self.hidden))
        if rec is not None:
            rec["logits256"].append(logits[0].to(torch.float16).cpu())
            rec["top8"].append(idx[0].cpu())
            rec["scores8"].append(scores[0].to(torch.float32).cpu())
        self._fill_experts(li, idx.view(-1).unique().tolist())
        out = lay.mlp(x)
        if isinstance(out, tuple):
            out = out[0]
        return residual + out
