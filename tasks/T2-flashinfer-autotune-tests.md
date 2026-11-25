# T2: FlashInfer autotune breaks bare-worker tests on SM100+

| Field | Value |
| --- | --- |
| Opened | [#11125](https://github.com/vllm-project/vllm-omni/pull/11125), 2026-11-24 |
| Status | Branch ready, results need a rerun on vLLM 0.34 |
| Branch | `fix-sm100-flashinfer-autotune-tests` (1 commit), base upstream/main `9b753874d` |
| Worktree | `.claude/worktrees/flashinfer-autotune-tests` |
| PR draft | `pr_desc/T2-flashinfer-autotune-tests.md` |
| Approved | 2026-11-15 |

## Problem

On an SM100+ host with FlashInfer installed, 15 CPU-marked tests fail on main.

- 14 tests build a bare `DiffusionWorker` with `object.__new__`.
  Since #9998, FlashInfer autotune is on by default on CUDA SM100+.
  The bare worker resolves it lazily, turns it on, then fails with `AttributeError: 'DiffusionWorker' object has no attribute 'rank'`.
  Hiding the GPU does not help: the capability check reads NVML.
- `test_backend_selection_reaches_the_dit_layer` (MammothModa2) selects `FLASH_ATTN`, which Blackwell rejects without FA4.

## Change

Test-only, 4 files, +26/-5.

- The bare-worker helpers set `_flashinfer_autotune_on = False`, which is what `load_model()` sets when autotune is off. #10346 fixed the same problem for the weight-sharing test.
- The MammothModa2 test resolves backend names from the registry, so it checks only the config-to-layer routing.
- Fixes 2 old mypy errors in `test_diffusion_step_pipeline.py`, because the local commit hook fails without them.
- A product-side fix was rejected: real workers always have `od_config` and `rank`.

## Results (upstream/main `b576ed001`, vLLM 0.33.0, RTX 5080)

| Run (4 changed files) | main | branch |
| --- | --- | --- |
| GPU visible | 15 failed, 109 passed | 124 passed |
| GPU hidden | 24 failed, 100 passed | 9 failed, 115 passed (same 9 fail on main) |
| pre-commit with mypy | - | pass |

## Rebase onto `9b753874d` (2026-11-23)

Clean.

## Open items

1. Rerun on vLLM 0.34, update the results, SHAs and versions in the PR body.

## Follow-ups (not in this PR)

- Whether SM120 consumer GPUs should turn on FlashInfer autotune by default is a question for the #9998 owners.
