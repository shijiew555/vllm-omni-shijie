# T1: SM120 host test failures (MiniMax-H3 offload, CUTLASS int8)

| Field | Value |
| --- | --- |
| Opened | [#11124](https://github.com/vllm-project/vllm-omni/pull/11124), 2026-11-24 |
| Status | Branch ready, results need a rerun on vLLM 0.34 |
| Branch | `fix-sm120-host-tests` (1 commit), base upstream/main `9b753874d` |
| Worktree | `.claude/worktrees/sm120-host-tests` |
| PR draft | `pr_desc/T1-sm120-host-tests.md` |
| Approved | 2026-11-14 |

## Problem

On a host with an SM120 GPU (RTX 5080 here), 14 CPU-marked tests fail on main.
CI has no SM120 runner, so upstream does not see them.

- 10 MiniMax-H3 tests (9 TeaCache extractor, 1 ControlNet) run the model on CPU.
  Since #7989 the model turns on AdaLN offload by default when the host GPU is SM120, and the offload needs CUDA.
- 4 int8 tests (1 int8 smoke, 3 ConvRot) call vLLM's CUTLASS int8 `scaled_mm`, which rejects int8 on SM100 and newer.
  vLLM's `CutlassInt8ScaledMMLinearKernel.is_supported` checks only for CUDA, so vLLM still selects it there.
  ConvRot calls `ops.cutlass_scaled_mm` directly.

## Change

Test-only, 4 files, +24 lines.

- The MiniMax-H3 test configs set `cache_config={"minimax_h3_adaln_offload": False}`.
- A per-file `cutlass_int8_supported` skip marker skips the CUTLASS int8 tests on CUDA capability 10 or higher.

## Results (upstream/main `b576ed001`, vLLM 0.33.0, RTX 5080)

| Run | main | branch |
| --- | --- | --- |
| CPU suite (config, quantization, cache, lora, offloader, minimax_h3) | 14 failed, 4090 passed, 14 skipped | 4100 passed, 18 skipped |
| Changed files, GPU visible | - | 122 passed, 10 skipped |
| pre-commit | - | pass |

## Rebase onto `9b753874d` (2026-11-23)

- Conflict with #10708, which moved diffusion INT8 onto vLLM's linear kernels and renamed the smoke test to `TestGPUInt8Smoke::test_real_gpu_int8_apply_forward`.
  Resolved: kept upstream's names and applied the skip to the renamed test.
- #10958 changed the MiniMax-H3 ControlNet and transformer code.

## Open items

1. Build a vLLM 0.34 venv and rerun the suite on main and on the branch.
2. Confirm the int8 and ConvRot failures still occur on SM120 after #10708. If they do not, drop that part of the PR and its title words.
3. Confirm the 10 MiniMax-H3 failures still occur after #10958.
4. Update the results, SHAs and versions in the PR body.

## Follow-ups (not in this PR)

- Production int8 and ConvRot could fail early with a clear error on SM100+.
- Report the `CutlassInt8ScaledMMLinearKernel.is_supported` selection bug to vLLM.
