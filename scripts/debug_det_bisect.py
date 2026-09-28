#!/usr/bin/env python3
"""Bisect first nonfinite under deterministic algorithms (prefill only)."""
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoTokenizer
from capture.stream_capture import StreamCapture

torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
mode = os.environ.get("DET_MODE", "full")
if mode in ("full", "warn"):
    torch.use_deterministic_algorithms(True, warn_only=(mode == "warn"))
    print(f"deterministic enabled, warn_only={mode == 'warn'}", flush=True)
else:
    print("deterministic OFF (baseline)", flush=True)

model = ROOT / "models/qwen-gptq-int4"
tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
text = "Explain why the sky is blue in one short paragraph:"
ids = tok.apply_chat_template([{"role": "user", "content": text}],
                              add_generation_prompt=True,
                              return_tensors="pt").input_ids[0].tolist()
print(f"prompt tokens: {len(ids)}", flush=True)

sc = StreamCapture(str(model))

# Instrument: wrap _step_mlp_residual and prefill loop manually with checks.
# Easiest: monkeypatch lay.mlp forward? Instead replicate prefill with checks.
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

m = sc.model.model
dev = sc.device
sc.cache = DynamicCache(config=sc.cfg)
ids_t = torch.tensor([list(ids)], device=dev)
h = m.embed_tokens(ids_t)
seq_len = h.shape[1]
text_pos = torch.arange(seq_len, device=dev).view(1, seq_len)
rot = m.rotary_emb(h, text_pos.expand(3, 1, seq_len))

def check(name, t, li):
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).sum().item()
        mx = t.detach().float().abs().max().item()
        print(f"NONFINITE {name} layer {li}: {bad} bad, max_abs={mx}", flush=True)
        return False
    return True

ok = True
with torch.no_grad():
    for li in range(sc.nlayers):
        lay = m.layers[li]
        sc._load_mixer_and_shared(li)
        residual_in = h
        x = lay.input_layernorm(h)
        if not check("input_layernorm_out", x, li):
            ok = False; break
        if sc.layer_types[li] == "linear_attention":
            x = lay.linear_attn(hidden_states=x, cache_params=sc.cache, attention_mask=None)
        else:
            mask = create_causal_mask(config=sc.cfg, inputs_embeds=x,
                                      attention_mask=None, past_key_values=sc.cache,
                                      position_ids=text_pos,
                                      layer_idx=lay.self_attn.layer_idx,
                                      allow_is_causal_skip=False)
            x, _ = lay.self_attn(hidden_states=x, attention_mask=mask,
                                 position_ids=text_pos, past_key_values=sc.cache,
                                 position_embeddings=rot)
        if not check("mixer_out", x, li):
            ok = False; break
        h_after_mix = residual_in + x
        if not check("h_after_mix", h_after_mix, li):
            ok = False; break
        router_input = lay.post_attention_layernorm(h_after_mix)
        logits, scores, idx = lay.mlp.gate(router_input.reshape(-1, sc.hidden))
        if not check("router_logits", logits.float(), li):
            ok = False; break
        sc._fill_experts(li, idx.reshape(-1).unique().tolist(), use_cache=False)
        out = lay.mlp(router_input)
        if isinstance(out, tuple):
            out = out[0]
        if not check("mlp_out", out, li):
            ok = False; break
        h = h_after_mix + out
        if not check("h_next", h, li):
            ok = False; break
        sc._drop_layer(li)
        if (li + 1) % 5 == 0:
            print(f"layers 0..{li} finite, h_absmax={h.float().abs().max().item():.1f}", flush=True)
    if ok:
        h = m.norm(h)
        logits = sc.model.lm_head(h[:, -1:, :])
        print(f"prefill logits finite: {bool(torch.isfinite(logits).all())} "
              f"max={logits.float().abs().max().item():.1f} argmax={int(logits.argmax())}", flush=True)
print("RESULT:", "ALL_FINITE" if ok else "FIRST_NONFINITE_FOUND", flush=True)
