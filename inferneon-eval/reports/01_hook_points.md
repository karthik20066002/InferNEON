# Qwen hook points — Phase B smoke

Status: `MEASURED` smoke validation only; full trace capture has not run.
Model: `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4`, revision
`00a66983516f8f8057741221277eed4141c9e431`.

The implementation intentionally calls the installed Transformers model
submodules rather than reimplementing the token mixer or MoE math.

## Layer order

From the installed
`transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`,
`Qwen3_5MoeDecoderLayer.forward`:

1. `input_layernorm`
2. `linear_attn` for `layer_types[L] == "linear_attention"`, otherwise
   `self_attn`
3. residual add
4. `post_attention_layernorm`
5. `mlp`, whose router is `mlp.gate`
6. second residual add

The source was inspected at the `Qwen3_5MoeDecoderLayer.forward` definition
(lines 945 onward in the installed 5.17.0 package). The model has 30
`linear_attention` layers and 10 `full_attention` layers at indices
3, 7, ..., 39.

## Saved tensors

`capture/stream_capture.py` records per processed token and layer:

- `h_in[L]`: hidden state at entry to the decoder layer, before
  `input_layernorm`.
- `h_postmix[L]`: `residual + token_mixer_output`, immediately before
  `post_attention_layernorm` and the MoE router.
- `router_input[L]`: the output of `post_attention_layernorm`, retained to
  audit the exact live router operation.
- `logits256[L]`, `top8[L]`, `scores8[L]`: outputs of
  `layer.mlp.gate` in router order.

The smoke validation recomputes `post_attention_layernorm(h_postmix[L])`, then
runs the same fp16 router linear and top-k operation. On the corrected smoke,
this matched 960/960 `(token, layer)` pairs (`1.000000`, `MEASURED`).

## Streaming weights

Only one layer's mixer/shared-expert weights are materialized at a time.
Routed GPTQ experts are dequantized only for the selected top-8 experts and
placed into the installed Qwen experts implementation's fused buffers.
Embeddings, final norm, routers, layer norms, shared gates, and LM head remain
resident. GPTQ qzero handling is documented in `capture/dequant.py` and was
checked against GPTQModel's official `TorchLinear` conversion.

This is a capture plumbing note, not a claim that the full Phase B dataset
has been captured.
