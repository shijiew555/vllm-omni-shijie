# T3: Runtime LoRA with online FP8 on Z-Image-Turbo

| Field | Value |
| --- | --- |
| Opened | [#11126](https://github.com/vllm-project/vllm-omni/pull/11126), 2026-11-24 |
| Status | Branch ready, results need a rerun on vLLM 0.34 |
| Branch | `lora-fp8-z-image-turbo` (2 commits), base upstream/main `9b753874d` |
| Worktree | `.claude/worktrees/lora-fp8-z-image-turbo` |
| PR draft | `pr_desc/T3-lora-fp8-z-image-turbo.md` |
| RFC | #1217, Feature x Feature table, LoRA Inference x FP8 Quant |
| Claim | [#1217 comment](https://github.com/vllm-project/vllm-omni/issues/1217#issuecomment-6891432700), 2026-11-15, no reply. The user decided the PR may open without a reply. |
| Approved | 2026-11-15 |

## Goal

Every LoRA cell in the #1217 matrix is unverified.
This task validates runtime PEFT LoRA with online FP8 quantization for Z-Image-Turbo, adds a CUDA test, and documents it.
It complements #9996 (LoRA with module-wise CPU offload, open) and does not touch the distill LoRA path of #10162.

## Change

- New test `tests/diffusion/lora/test_diffusion_lora_fp8.py::test_zimage_lora_online_fp8` (separate file, because #9996 rewrites `test_diffusion_lora.py`).
  Two synthetic adapters: rank-8 fused `to_qkv` (as in #9996), and rank-16 split Q/K/V, `to_out.0`, `w1/w2/w3` and the unquantized `adaLN_modulation.0`.
  Requests: base, A, B, A, scale 0, A, base, B, run twice, all exact comparisons.
  Marks match #9996, so the L4 merge job collects it.
- Docs: a "Runtime LoRA with online FP8 quantization" section in the LoRA guide and the LoRA x FP8 matrix cell.
- No production code changes. No bug found.

## Results (upstream/main `b576ed001`, vLLM 0.33.0, RTX 5080 16 GB, host RAM 30 GB)

| Check | Result |
| --- | --- |
| FP8 test (needs `expandable_segments:True` on 16 GB) | 1 passed, 36.5 s; peak 12.1 GB during requests, about 15.4 GB at startup |
| CPU LoRA tests, GPU visible | 176 passed |
| pre-commit | pass |
| FP8 with a real pixel-art adapter, 1024x1024, 8 steps | median 2.83 s base, 3.09 s with adapter; peak 14.4 GB; style clearly applied |

The FP8 vs BF16 comparison was dropped: BF16 does not fit 16 GB, and its module-offload path used about 28 GB of the 30 GB host RAM and crashed the machine 3 times.

## Rebase onto `9b753874d` (2026-11-23)

Clean.

## Open items

1. Rerun the FP8 test on vLLM 0.34, under `systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0`.
2. Update the results, SHAs and versions in the PR body.
3. Never run a BF16 or offload Z-Image-Turbo job on this machine.

## PR activity

- 2026-11-24: posted the self-review requested by vllm-omni-review-bot. The Codex review did not run (repository usage limit), so it has no findings.
