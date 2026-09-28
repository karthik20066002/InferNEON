# RESUME — rewritten 2026-09-28 after restart (old file was NUL-corrupted by power-off)

## Intact, do NOT redo
- `configs/default.yaml` (3 calib / 4 heldout, all defaults, median+mean)
- `data/prompts/` 35 prompts (7/domain x 5 domains, split verified)
- `reports/00_environment.md` (Phase A MEASURED: 4070 8GB, Ryzen 9 8945HS, 140G avail, net ONLINE, qwen3_5_moe needs transformers>=5.17, rev 995ad96e, 71.9GB)
- `reports/questions_for_helen.md`, `capture/README.md`
- `.venv` Python 3.12.13 + transformers 5.17.0 + hub + numpy (MEASURED)
- uv cache ~1.5G partial CUDA wheels (resumable)

## Pending at restart
- ~~torch + accelerate NOT INSTALLED~~ RESOLVED 2026-09-28 (MEASURED): torch
  2.14.0+cu130 + CUDA True on RTX 4070 Laptop, accelerate 1.15.0, transformers
  5.17.0, safetensors + numpy present. Q4 (torch channel) answered by measurement.
- gptqmodel / quant libs NOT INSTALLED
- No model weights downloaded (only small JSON). Approved primary: `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4`, verify router dtypes, device_map=auto + disk offload
- Disk now 109G avail (was 140G; OLMoE weights + 12G firmware image on /home)
- Next gates: gptqmodel → pin palmfuture rev (Helen Q1) → 20GB download approval → 1-prompt capture + validate → full capture → sim+tests → policies/helen → E1,E3...

## Resume commands
- `tail -f /tmp/torch_resume.log`
- `.venv/bin/python -c "import torch,accelerate; print(torch.__version__, torch.cuda.is_available())"`
- Then: `uv pip install --python .venv/bin/python gptqmodel`
- Then pin rev: record `palmfuture` commit hash before `hf_hub_download` / `from_pretrained` with `device_map=auto`.
