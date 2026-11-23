# PR drafts

Each folder holds the PR title (`title.txt`) and the exact PR body (`body.md`).
None of these PRs is opened yet.
None of them needs a separate issue: T1 and T2 are small test fixes, and T3 links its roadmap issue #1217.

| Task | Title | Branch | Head | Base |
| --- | --- | --- | --- | --- |
| T1 | `[Bugfix][CI] Keep MiniMax-H3 CPU and int8 CUDA tests independent of SM120 hosts` | `fix-sm120-host-tests` | `76aea933c` | upstream/main `9b753874d` |
| T2 | `[Bugfix][CI/Build] Keep diffusion CPU unit tests host-independent on SM100+ GPUs` | `fix-sm100-flashinfer-autotune-tests` | `8bb727a72` | upstream/main `9b753874d` |
| T3 | `[Model] Validate Z-Image-Turbo runtime LoRA with online FP8 quantization` | `lora-fp8-z-image-turbo` | `f190ff50e` (2 commits) | upstream/main `9b753874d` |

## Status (2026-11-23)

All three branches are rebased onto upstream/main `9b753874d` and pass `pre-commit` on their changed files.

The test results in the bodies were measured on upstream/main `b576ed001` with vLLM 0.33.0.
Upstream main now needs vLLM 0.34 (#10365), so the results must be measured again before posting:

- T1: rebasing conflicted with #10708, which moved diffusion INT8 onto vLLM's linear kernels and renamed the smoke test to `TestGPUInt8Smoke::test_real_gpu_int8_apply_forward`.
  The SM100+ skip now applies to the renamed test.
  Check on vLLM 0.34 that the INT8 and ConvRot failures still occur on SM120; if they do not, drop that part.
  #10958 changed the MiniMax-H3 ControlNet and transformer code; check that the 10 MiniMax-H3 failures still occur.
- T2: rebased without conflicts. Measure the 15 failures again.
- T3: rebased without conflicts. Run the FP8 test again, under a 20 GB host memory cap.
  Never run the BF16 offload reference on this machine: it exhausts the 30 GB of host RAM.
  The claim on #1217 has no reply yet; the PR may open without one.

After the new measurements, update the commit SHAs, the vLLM version and the vLLM-Omni commit in each body.

## Posting

```bash
gh pr create -R vllm-project/vllm-omni --base main --head shijiew555:<branch> \
  --title "$(cat pr_desc/<task>/title.txt)" --body-file pr_desc/<task>/body.md
```
