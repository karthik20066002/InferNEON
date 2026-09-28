#!/usr/bin/env python3
"""Smoke: 1 prompt, N decode tokens, streaming layer weights. Phase B gate.

Prints per-layer top-8, runs the §4.4 offline recompute (top-8 from saved
h_postmix + saved router must match), and reports MATCH RATE.
"""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, torch
from transformers import AutoTokenizer
from capture.stream_capture import StreamCapture

MODEL = 'models/qwen-gptq-int4'
PROMPT = 'Explain why the sky is blue in one short paragraph:'
NTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 5
DEV = 'cuda'
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
t0 = time.time()
sc = StreamCapture(MODEL, device=DEV)
print('SETUP %.1fs | layers=%d' % (time.time() - t0, sc.nlayers), flush=True)
print('VRAM alloc %.2f GiB reserved %.2f GiB' % (
    torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30), flush=True)

# The brief requires the model chat template. The logits returned while
# processing the final prompt token are the distribution for decode token 0;
# do not feed that prompt token a second time.
chat = tok.apply_chat_template(
    [{'role': 'user', 'content': PROMPT}],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors='pt',
)
ids = chat.input_ids[0].tolist()
print('prompt tokens:', len(ids), flush=True)

recs = []
generated = []
t0 = time.time()
with torch.no_grad():
    prompt_records = [
        {'h_in': [], 'h_postmix': [], 'router_input': [], 'top8': [], 'scores8': [], 'logits256': []}
        for _ in ids
    ]
    _, prompt_logits = sc.prefill(ids, prompt_records)
    recs.extend(('prefill', i, r) for i, r in enumerate(prompt_records))
    print('prefill %.1fs | VRAM peak %.2f GiB' % (
        time.time() - t0, torch.cuda.max_memory_allocated() / 2**30), flush=True)
    cur = int(torch.argmax(prompt_logits))
    for j in range(NTOK):
        generated.append(cur)
        r = {'h_in': [], 'h_postmix': [], 'router_input': [], 'top8': [], 'scores8': [], 'logits256': []}
        _, logits = sc.step(cur, r, pos=len(ids) + j)
        cur = int(torch.argmax(logits))
        recs.append(('decode', j, r))
        print('decode %d -> token %d  L0 top8=%s  L39 top8=%s' % (
            j, generated[-1], recs[-1][2]['top8'][0].tolist(), recs[-1][2]['top8'][39].tolist()), flush=True)
        print('  %.1fs/token | VRAM peak %.2f GiB' % (
            (time.time() - t0) / (j + 1), torch.cuda.max_memory_allocated() / 2**30), flush=True)
print('OUT:', repr(tok.decode(generated, skip_special_tokens=True)), flush=True)

# §4.4 offline recompute. Reapply the saved pre-router norm to the saved
# h_postmix, then use the same fp16 router operation as the live path.
# Prefill rows are recomputed as one batch because the live prefill is one
# batch; decode rows remain one-token operations.
tot = ok = 0
for L in range(sc.nlayers):
    for phase in ('prefill', 'decode'):
        group = [r for p, _, r in recs if p == phase]
        if not group:
            continue
        h = torch.stack([r['h_postmix'][L] for r in group]).to(DEV)
        normed = sc.model.model.layers[L].post_attention_layernorm(h.view(-1, 1, sc.hidden))
        w = sc.model.model.layers[L].mlp.gate.weight
        lg = torch.nn.functional.linear(normed.view(-1, sc.hidden), w)
        for row, r in zip(lg, group):
            tk = torch.topk(row, 8).indices
            ok += int(tk.tolist() == r['top8'][L].tolist()); tot += 1
print('RECOMPUTE MATCH: %d/%d = %.6f' % (ok, tot, ok / tot), flush=True)
assert ok / tot >= 0.999, 'router recompute validation below 99.9%'

# shape/range checks
r = recs[-1][2]
assert all(len(t) == 8 and len(set(t)) == 8 and 0 <= min(t) and max(t) < 256 for t in r['top8'])
assert all(torch.isfinite(s).all() for s in r['scores8'])
assert all(l.shape == (256,) for l in r['logits256'])
print('SHAPE CHECKS OK', flush=True)
