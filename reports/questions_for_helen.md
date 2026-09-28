# Questions for Helen — decisions only you make

> All Q1–Q9 resolved 2026-09-28 via the questions tool; resolutions in
> CHANGELOG.md. New questions append below.

1. Quant fork lock: primary `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` at which exact rev? Fallback order if router turns out quantized or kernels fail on RTX 4070 laptop (Ada 8.9)?
2. `configs/default.yaml` split 3 calib / 4 heldout confirmed. Keep 512-token minimum, greedy, batch 1, seed 0?
3. Reporting: median+mean confirmed. Keep P50/P95 for E4 plus max/P95 added delay, and windowed hit-rate plots for E2?
4. Torch CUDA channel: default pip `torch` (cu128-class) OK, or pin cu124/cu126 for driver 615.71.09 / CUDA 13.4?
5. Disk offload path for 20GB GPTQ + 8GB VRAM / 6.5GB avail RAM: allow `HF_HOME` + offload folder under `/home` (140G avail)? Cap?
6. `policies/helen/` signatures: prefer abstract classes vs Protocols for the 5 modules? Stub names OK?
7. E7 scope: if GPTQ-vs-BF16 agreement needs BF16 71.9GB touch, allow partial (few hundred tokens, few prompts) or declare impossible for now?
8. Init git in `inferneon-eval/` for commit hashes in provenance? Currently no repo.
9. Trace-capture throughput (MEASURED 2026-09-28): OLMoE BF16 smoke run did 1.1 tok/s
   under CPU/disk offload (64 tokens in 60.3s). At that rate the brief's minimum
   capture (35 prompts x 512 decode tokens, plus prefill + hook overhead + hidden
   states) is roughly 5+ hours of wall time, before the Qwen GPTQ path is even
   measured. Options: (a) run capture unattended overnight as-is, (b) trim
   per-prompt token minimums or held-out hidden-state coverage, (c) accept the
   time and start with the 1-prompt validation first. Which?
10. Qwen load blocked by RAM (MEASURED 2026-09-28): 3 attempts, EXIT=137 every
    time, journal peaks anon-rss 8.3 / 10.9 / 11.6 GB vs 14 GB RAM + 8 GB swap
    (2.8 used). Dies right after weights stream, before dispatch. MTP (1.57 GB)
    + vision (0.83 GB) are dead weight for text-only work but the loader cannot
    skip them without surgery. Desktop already lean (~2.5 GB). Options:
    (a) MTP+vision loader surgery (text-path-identical, saves ~2.4 GB, may still
    fail), (b) validate hooks/sim/harness on OLMoE-BF16 now (fits, runs) and do
    Qwen traces when RAM allows, (c) stop and wait for hardware change. Which?
