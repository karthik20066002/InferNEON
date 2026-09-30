#!/usr/bin/env python3
"""olmoe_host_ref.py — host ground truth for the firmware generation proof.

Runs the REAL transformers OlmoeForCausalLM so the firmware output has an
independent reference: if both agree on the generated token ids, a shared
conceptual bug in my Rust forward is ruled out. No math is reimplemented here.

Memory: the checkpoint is 13.8 GB bf16, of which only ~950 MB is non-expert
(embed + attn + head). A token touches 8 of 64 experts per layer = 128 of the
1024 experts, i.e. ~1.6 GB of the 12.6 GB. So: instantiate the model under
init_empty_weights, materialise the small weights eagerly, and let each
OlmoeExperts block page in its own expert rows from the safetensors mmap the
moment its layer runs. The page cache is the cache; peak resident set is
~2 GB, so no VRAM budget, no accelerate offload folder, no disk spilling.

Emits JSON: prompt ids, generated ids, per-step top-10 probs and top-2 logit
gap, and per-layer top-8 routed expert sets + weights. The route dump is the
fine-grained comparison: agreement there localises any divergence to a layer
rather than just "the text differs".
"""
import argparse
import json
import os
import sys

import torch
from safetensors import safe_open


def build_lazy_experts(model, handles, name_to_handle, dtype, device):
    """Wrap each OlmoeExperts.forward to page its rows in from mmap, then drop.

    Only the experts this token actually routes to are ever read. After the
    layer finishes, the 3D stacks are released again so resident memory stays
    at one layer's working set (64 x 12.6 MB worst case, 8 x 12.6 MB typical).
    """
    from transformers.models.olmoe.modeling_olmoe import OlmoeExperts

    orig_forward = OlmoeExperts.forward
    stats = {'layers_paged': 0, 'expert_rows_read': 0}

    def page_in(module: OlmoeExperts, layer_idx: int, top_k_index: torch.Tensor):
        needed = sorted({int(e) for e in top_k_index.reshape(-1).tolist()})
        gu = torch.empty(module.num_experts, 2 * module.intermediate_dim,
                         module.hidden_dim, dtype=dtype, device=device)
        dn = torch.empty(module.num_experts, module.hidden_dim,
                         module.intermediate_dim, dtype=dtype, device=device)
        for e in needed:
            p = f'model.layers.{layer_idx}.mlp.experts.{e}.'
            gate = name_to_handle[p + 'gate_proj.weight'].get_tensor(p + 'gate_proj.weight')
            up = name_to_handle[p + 'up_proj.weight'].get_tensor(p + 'up_proj.weight')
            down = name_to_handle[p + 'down_proj.weight'].get_tensor(p + 'down_proj.weight')
            gu[e, :module.intermediate_dim] = gate
            gu[e, module.intermediate_dim:] = up
            dn[e] = down
        module.gate_up_proj = torch.nn.Parameter(gu, requires_grad=False)
        module.down_proj = torch.nn.Parameter(dn, requires_grad=False)
        stats['layers_paged'] += 1
        stats['expert_rows_read'] += len(needed)
        return needed

    def patched_forward(self, hidden_states, top_k_index, top_k_weights):
        li = int(self._layer_idx)
        page_in(self, li, top_k_index)
        out = orig_forward(self, hidden_states, top_k_index, top_k_weights)
        # Release: the mmap is the storage of record; page cache keeps it cheap.
        self.gate_up_proj = torch.nn.Parameter(
            torch.empty(0, dtype=dtype, device=device), requires_grad=False)
        self.down_proj = torch.nn.Parameter(
            torch.empty(0, dtype=dtype, device=device), requires_grad=False)
        return out

    OlmoeExperts.forward = patched_forward
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default='models/olmoe-1b-7b')
    ap.add_argument('--prompt-ids', default='510,8467,310,4797')
    ap.add_argument('--new-tokens', type=int, default=3)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='reports/olmoe_host_ref.json')
    args = ap.parse_args()

    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.olmoe.modeling_olmoe import OlmoeExperts

    ids = [int(x) for x in args.prompt_ids.split(',')]
    device = args.device if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16
    print(f'prompt ids: {ids}  device: {device}  dtype: {dtype}', flush=True)

    # Open every shard once; safetensors mmaps, so no weight bytes are read
    # until get_tensor touches them.
    shards = sorted({f for f in os.listdir(args.model_dir) if f.endswith('.safetensors')})
    handles = {s: safe_open(os.path.join(args.model_dir, s), framework='pt') for s in shards}
    name_to_handle = {}
    for s, h in handles.items():
        for k in h.keys():
            name_to_handle[k] = h
    print(f'{len(shards)} shards, {len(name_to_handle)} tensors (mmapped, not loaded)', flush=True)

    config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
    config.output_router_logits = True
    config.norm_topk_prob = getattr(config, 'norm_topk_prob', False)
    rope = dict(getattr(config, 'rope_parameters', None) or {})
    print(f'norm_topk_prob={config.norm_topk_prob} top_k={config.num_experts_per_tok} '
          f'experts={config.num_experts} rope_theta={rope.get("rope_theta")} '
          f'rope_type={rope.get("rope_type")}', flush=True)

    # Shell with no storage, then fill the ~950 MB of non-expert weights.
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
    model.eval()

    # NOTE: no model.to(device) anywhere. The expert stacks stay on meta until
    # the pager fills them, and .to() would try to copy meta (NotImplemented).
    # Every tensor is placed on `device` at fill time instead.
    small = 0
    with torch.no_grad():
        for name in list(name_to_handle):
            if '.experts.' in name or not name.endswith('.weight'):
                continue
            try:
                mod = model.get_submodule(name[: -len('.weight')])
            except AttributeError:
                continue
            t = name_to_handle[name].get_tensor(name).to(device=device, dtype=dtype)
            mod.weight = torch.nn.Parameter(t, requires_grad=False)
            small += t.numel() * t.element_size()
    print(f'non-expert weights resident: {small / 2**20:.0f} MiB on {device}', flush=True)

    for li, layer in enumerate(model.model.layers):
        layer.mlp.experts._layer_idx = li
    assert issubclass(type(model.model.layers[0].mlp.experts), OlmoeExperts)
    stats = build_lazy_experts(model, handles, name_to_handle, dtype, device)

    # Rotary buffers are real (not meta) under init_empty_weights, but pin the
    # module to the device so cos/sin land next to the activations.
    rot = model.model.rotary_emb
    for name in ('inv_freq', 'original_inv_freq', 'cos_cached', 'sin_cached'):
        buf = getattr(rot, name, None)
        if buf is not None:
            setattr(rot, name, buf.to(device))

    result = {
        'model_dir': os.path.abspath(args.model_dir),
        'prompt_ids': ids,
        'dtype': str(dtype),
        'device': device,
        'torch': torch.__version__,
        'transformers': __import__('transformers').__version__,
        'norm_topk_prob': bool(config.norm_topk_prob),
        'non_expert_mib': round(small / 2**20, 1),
        'steps': [],
    }

    def routes(rl):
        rows = []
        for ll in rl:
            t = ll[0] if isinstance(ll, tuple) else ll
            while t.dim() > 1:  # (layers, seq, experts) -> (seq, experts) -> (experts,)
                t = t[-1] if t.dim() == 2 else t[0]
            w = torch.softmax(t.float(), dim=-1)
            v, i = torch.topk(w, config.num_experts_per_tok)
            rows.append({'ids': [int(x) for x in i.tolist()],
                         'w': [round(float(x), 5) for x in v.tolist()]})
        return rows

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(input_ids=input_ids, use_cache=True, output_router_logits=True)
    past = out.past_key_values
    logits = out.logits[0, -1].float()
    result['prompt_routes'] = routes(out.router_logits) if out.router_logits else None

    n = 0
    while n < args.new_tokens:
        probs = torch.softmax(logits, dim=-1)
        topv, topi = torch.topk(probs, 10)
        nxt = int(torch.argmax(logits))
        t2 = torch.topk(logits, 2).values
        step = {
            'in': int(input_ids[0, -1]),
            'next': nxt,
            'top10_ids': [int(x) for x in topi.tolist()],
            'top10_p': [round(float(x), 6) for x in topv.tolist()],
            'top10_logits': [round(float(logits[topi[k]]), 5) for k in range(10)],
            'logit_max': round(float(logits.max()), 5),
            'logit_argmax': round(float(logits[nxt]), 5),
            'logit_top2_gap': round(float(t2[0] - t2[1]), 5),
        }
        result['steps'].append(step)
        print(f'step {n}: in={step["in"]} next={nxt} top5={step["top10_ids"][:5]} '
              f'gap={step["logit_top2_gap"]}', flush=True)
        if nxt == 50279:
            print('EOS', flush=True)
            break
        input_ids = torch.tensor([[nxt]], dtype=torch.long, device=device)
        out = model(input_ids=input_ids, past_key_values=past, use_cache=True,
                    output_router_logits=True)
        past = out.past_key_values
        logits = out.logits[0, -1].float()
        if out.router_logits:
            result.setdefault('decode_routes', []).append(routes(out.router_logits))
        n += 1

    result['generated_ids'] = [s['next'] for s in result['steps']]
    result['lazy_stats'] = stats
    print('generated ids:', result['generated_ids'], flush=True)
    print('lazy paging:', stats, flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
