#!/usr/bin/env python3
"""Inner bisect of layer-3 full attention under deterministic mode."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    apply_rotary_pos_emb,
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)
from capture.stream_capture import StreamCapture

torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.use_deterministic_algorithms(True)
print("deterministic on", flush=True)

model = ROOT / "models/qwen-gptq-int4"
tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
text = "Explain why the sky is blue in one short paragraph:"
ids = tok.apply_chat_template([{"role": "user", "content": text}],
                              add_generation_prompt=True,
                              return_tensors="pt").input_ids[0].tolist()

sc = StreamCapture(str(model))
m = sc.model.model
dev = sc.device
sc.cache = DynamicCache(config=sc.cfg)
ids_t = torch.tensor([list(ids)], device=dev)
h = m.embed_tokens(ids_t)
seq_len = h.shape[1]
text_pos = torch.arange(seq_len, device=dev).view(1, seq_len)
rot = m.rotary_emb(h, text_pos.expand(3, 1, seq_len))


def ck(name, t):
    f = bool(torch.isfinite(t).all())
    try:
        mx = float(t.detach().float().abs().max())
    except Exception:
        mx = float("nan")
    print(f"  {name}: finite={f} absmax={mx} dtype={t.dtype} shape={tuple(t.shape)}", flush=True)
    return f


with torch.no_grad():
    for li in range(3):
        lay = m.layers[li]
        sc._load_mixer_and_shared(li)
        residual_in = h
        x = lay.input_layernorm(h)
        x = lay.linear_attn(hidden_states=x, cache_params=sc.cache, attention_mask=None)
        h_after_mix = residual_in + x
        router_input = lay.post_attention_layernorm(h_after_mix)
        logits, scores, idx = lay.mlp.gate(router_input.reshape(-1, sc.hidden))
        sc._fill_experts(li, idx.reshape(-1).unique().tolist(), use_cache=False)
        out = lay.mlp(router_input)
        if isinstance(out, tuple):
            out = out[0]
        h = h_after_mix + out
        sc._drop_layer(li)
    print(f"layers 0-2 done, h finite={bool(torch.isfinite(h).all())}", flush=True)

    li = 3
    lay = m.layers[li]
    sc._load_mixer_and_shared(li)
    print(f"layer 3 type: {sc.layer_types[li]}", flush=True)
    residual_in = h
    x = lay.input_layernorm(h)
    ck("input_layernorm", x)
    attn = lay.self_attn
    mask = create_causal_mask(config=sc.cfg, inputs_embeds=x, attention_mask=None,
                              past_key_values=sc.cache, position_ids=text_pos,
                              layer_idx=attn.layer_idx, allow_is_causal_skip=False)
    ck("causal_mask", mask.float() if mask is not None else torch.tensor(float("nan")))
    print(f"  mask dtype={None if mask is None else mask.dtype} shape={None if mask is None else tuple(mask.shape)}", flush=True)
    hs = x.shape[:-1]
    hidden_shape = (*hs, -1, attn.head_dim)
    qg = attn.q_proj(x).view(*hs, -1, attn.head_dim * 2)
    ck("q_proj", qg)
    query_states, gate = torch.chunk(qg, 2, dim=-1)
    ck("q_chunk", query_states)
    ck("gate_chunk", gate)
    gate_r = gate.reshape(*hs, -1)
    query_states = attn.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    ck("q_norm", query_states)
    key_states = attn.k_norm(attn.k_proj(x).view(hidden_shape)).transpose(1, 2)
    ck("k_norm", key_states)
    value_states = attn.v_proj(x).view(hidden_shape).transpose(1, 2)
    ck("v_proj", value_states)
    cos, sin = rot
    # rot was built for embed; attention gets T row like step(); in prefill
    # StreamCapture passes full `rot` tuple to self_attn.
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    ck("q_rope", query_states)
    ck("k_rope", key_states)
    key_states, value_states = sc.cache.update(key_states, value_states, attn.layer_idx)
    ck("k_cached", key_states)
    ck("v_cached", value_states)
    interface = ALL_ATTENTION_FUNCTIONS.get_interface(sc.cfg._attn_implementation
                                                      if hasattr(sc.cfg, "_attn_implementation") else "eager",
                                                      eager_attention_forward)
    print(f"  attn interface: {interface}", flush=True)
    attn_output, attn_weights = interface(attn, query_states, key_states, value_states,
                                          mask, dropout=0.0, scaling=attn.scaling)
    ck("attn_out", attn_output)
    attn_output = attn_output.reshape(*hs, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate_r)
    ck("gated", attn_output)
    attn_output = attn.o_proj(attn_output)
    ck("o_proj", attn_output)
print("DONE", flush=True)
