#!/usr/bin/env python3
"""Real-model rotary_emb under nondet vs det. Init once, no layers run."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoTokenizer
from capture.stream_capture import StreamCapture

torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False

model = ROOT / "models/qwen-gptq-int4"
tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
text = "Explain why the sky is blue in one short paragraph:"
ids = tok.apply_chat_template([{"role": "user", "content": text}],
                              add_generation_prompt=True,
                              return_tensors="pt").input_ids[0].tolist()

sc = StreamCapture(str(model))
m = sc.model.model
dev = sc.device
print(f"inv_freq is_meta={m.rotary_emb.inv_freq.is_meta} "
      f"shape={tuple(m.rotary_emb.inv_freq.shape)} "
      f"device={m.rotary_emb.inv_freq.device}", flush=True)
ids_t = torch.tensor([list(ids)], device=dev)
with torch.no_grad():
    h = m.embed_tokens(ids_t)
seq_len = h.shape[1]
text_pos = torch.arange(seq_len, device=dev).view(1, seq_len)
with torch.no_grad():
    c0, s0 = m.rotary_emb(h, text_pos.expand(3, 1, seq_len))
print(f"nondet: cos is_meta={c0.is_meta} sin is_meta={s0.is_meta} "
      f"shapes={tuple(c0.shape)},{tuple(s0.shape)}", flush=True)
if not c0.is_meta:
    print(f"nondet: cos finite={bool(torch.isfinite(c0).all())} sin finite={bool(torch.isfinite(s0).all())}", flush=True)
    print(f"nondet: cos mean={float(c0.float().mean())} std={float(c0.float().std())} "
          f"frac_cos_eq_1={float((c0==1).float().mean())} frac_sin_eq_0={float((s0==0).float().mean())} "
          f"cos_absmax={float(c0.float().abs().max())}", flush=True)
torch.use_deterministic_algorithms(True)
with torch.no_grad():
    c1, s1 = m.rotary_emb(h, text_pos.expand(3, 1, seq_len))
print(f"det:    cos is_meta={c1.is_meta} sin is_meta={s1.is_meta}", flush=True)
if not c1.is_meta and torch.isfinite(c1).all() and torch.isfinite(s1).all():
    print(f"det==nondet: cos={bool(torch.equal(c0, c1))} sin={bool(torch.equal(s0, s1))}", flush=True)
print("DONE", flush=True)
