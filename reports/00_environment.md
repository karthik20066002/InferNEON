# 00_environment — Phase A (MEASURED unless marked)

## Post-restart correction (MEASURED 2026-09-28)

The earlier pre-install statements below are historical and stale. Current eval
venv measurements are: Python 3.12.13, torch 2.14.0+cu130 with CUDA available,
accelerate 1.15.0, transformers 5.17.0, gptqmodel 7.5.0, optimum installed,
safetensors 0.8.0, and numpy 2.5.3. GPU is an NVIDIA GeForce RTX 4070 Laptop
GPU with 8188 MiB VRAM, driver 615.71.09. The local pinned GPTQ model is present
under `models/qwen-gptq-int4`.

The first whole-model `transformers`/GPTQ dispatch path was attempted three
times and was killed by the kernel OOM killer after shard loading (exit 137).
The current capture path therefore streams one decoder layer at a time and
materializes only selected routed experts. A corrected 22-prompt-token +
2-decode-token smoke completed with 3.59 GiB peak VRAM and no non-finite
activations (`MEASURED`).

All numbers below are `MEASURED` from runs on this machine on 2026-09-27 unless marked `ASSUMED`. Failures reported plainly.

## GPU / driver / CUDA — MEASURED
- `nvidia-smi`: NVIDIA GeForce RTX 4070 Laptop GPU, 8188 MiB, driver 615.71.09, compute_cap 8.9
- `nvcc`: Cuda compilation tools release 13.4, V13.4.92
- VRAM note: 8 GiB will NOT fit BF16 71.9 GB. Quantized + CPU/disk offload required. Slow is acceptable per brief.

## CPU / RAM / disk — MEASURED
- CPU: AMD Ryzen 9 8945HS w/ Radeon 780M, 8C/16T, x86_64
- RAM: total 14 Gi, available ~6.5 Gi at check time (`free -h`)
- Disk: /home 344G total, 140G avail (58% used). /tmp tmpfs 7.5G.
- Swap: 7.9 Gi

## Python / libs — MEASURED
- System: Python 3.14.7 (`/usr/bin/python3`), torch NOT INSTALLED, transformers NOT INSTALLED system-wide
- Eval venv: `/home/axiss/Documents/projects/InferNEON/inferneon-eval/.venv`, Python 3.12.13
- In venv: transformers 5.17.0 MEASURED, huggingface_hub 1.33.0 MEASURED, numpy 2.5.3 MEASURED, safetensors 0.8.0, tokenizers 0.23.2
- NOT INSTALLED in venv (MEASURED absence): torch, accelerate, auto-gptq / gptqmodel / awq / optimum. Needed before capture.
- No git repo in `Documents/projects/InferNEON/` (MEASURED: `fatal: not a git repository`). Reproducibility via config hash + model rev until repo init decided.

## Transformers support for `qwen3_5_moe` — MEASURED
- `transformers==4.57.1` (version stamped in upstream config.json): FAILS. `KeyError: 'qwen3_5_moe'`, `models/qwen3_5_moe` absent. Only `qwen2*`, `qwen3*`, `qwen3_moe`, `qwen3_next` present.
- `transformers==5.17.0` (PyPI latest at check): SUPPORTS. `has qwen3_5_moe: True`, `AutoConfig.from_pretrained('Qwen/Qwen3.6-35B-A3B')` → `Qwen3_5MoeConfig`, `model_type=qwen3_5_moe` MEASURED.
- Upstream repo ships NO custom modeling code (MEASURED: siblings are config/tokenizer/safetensors only, `configuration.json` is `{"framework":"Pytorch","task":"visual-question-answering"}`). So stock transformers >=5.17 required. Pin to 5.17.0 in env.

## Internet — MEASURED ONLINE
- `https://huggingface.co` → 200 via curl + urllib. Tavily-style proxy block not hit via built-in path. Offline stop NOT triggered.

## Model config vs §1 — MEASURED, matches
From `AutoConfig` on rev `995ad96eacd98c81ed38be0c5b274b04031597b0`:
- 40 layers MEASURED; `layer_types` 3x linear_attention + 1 full_attention repeating; full indices [3,7,11,15,19,23,27,31,35,39] (10 full) MEASURED
- 256 routed experts, top-8, hidden 2048, moe_inter 512, shared_inter 512 MEASURED
- heads 16Q / 2KV, head_dim 256, vocab 248320, tie_word_embeddings False MEASURED
- `mtp_num_hidden_layers` 1 MEASURED (record loaded, unused per brief)
- `model.safetensors.index.json` total_size 71903645408 (~71.9 GB) MEASURED
- Derived (ASSUMED dtype BF16 for math, to confirm after load): routed expert 3*2048*512=3,145,728 weights ≈ 6.3 MB BF16 / ~1.5 MiB INT4; LM head 2048*248320 ≈ 509M params; full-attn KV 10*2*256*2*2 = 20,480 B/token → ~80 MiB at 4k. Compute-side ASSUMED until weighed.
- Multimodal + MTP present in checkpoint per config (vision_config + mtp). Study is text-only per brief. Will record whether MTP weights load.

## Quantized candidates — MEASURED listing, router dtype UNCONFIRMED
- `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4`: 6 safetensors shards + `quantize_config.json` (bits 4, group 128, gptqmodel 6.0.3). Dynamic exclusions observed include `-:.*mlp\.gate$`, `-:.*attn.*`, `-:.*shared_expert.*`, `-:.*mtp.*`, `lm_head:false` — suggests router/gate kept out of quant, TO CONFIRM via tensor dtypes after download. Size ~20 GB class ASSUMED until weighed.
- `nvidia/Qwen3.6-35B-A3B-NVFP4`: 3 shards, ModelOpt MIXED_PRECISION (experts W4A16_NVFP4 group 16, attn FP8). Likely needs ModelOpt/TRT-LLM kernels, risk on plain transformers + 4070 laptop. Fallback only.
- `Qwen/Qwen3.6-35B-A3B-FP8`: 56 per-layer shards. ~35 GB class ASSUMED. Exceeds comfortable RAM+VRAM without aggressive offload.
- BF16 upstream (26 shards, 71.9 GB): NOT loadable on 8 GB VRAM / 14 GB RAM without full disk offload. Reference only for routing-agreement spot check if resources allow.

## Proposed loading plan — needs Helen approval before big download
1. `uv pip install torch (cu124/cu128) + accelerate + gptqmodel` in `.venv`. ~3-5 GB download ASSUMED. Then confirm `torch.cuda.is_available()` MEASURED.
2. Download `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` at pinned rev (record hash), `device_map=auto` with max-memory VRAM ~7GB + CPU + disk offload. Verify: tensor dtypes (router/gate unquantized?), `model_type`, module tree print, routing code path.
3. Small routing-agreement check later (Phase A/E7): few hundred tokens vs higher-precision reference if RAM/disk allow, else explicitly report impossible.
4. If GPTQ router turns out quantized or kernels fail on Ada laptop, fallback order: `btbtyler09/GPTQ-4bit` → `Qwen/FP8` with stronger offload → report.
5. Do NOT download BF16 71.9 GB now (disk allows but RAM/VRAM do not).

## Stop / wait
Per brief §8.1: stopping here. No model weights downloaded yet (only config.json + small json via API, <1 MB). Awaiting Helen approval for step 1-2 above. Questions in `reports/questions_for_helen.md` (to be written next if requested).
