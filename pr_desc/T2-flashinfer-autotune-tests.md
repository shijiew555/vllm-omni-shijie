# T2: [Bugfix][CI/Build] Keep diffusion CPU unit tests host-independent on SM100+ GPUs

| Field | Value |
| --- | --- |
| Type | Pull request |
| Title | `[Bugfix][CI/Build] Keep diffusion CPU unit tests host-independent on SM100+ GPUs` |
| Open the PR | [Open PR in one click](https://github.com/vllm-project/vllm-omni/compare/main...shijiew555:vllm-omni-shijie:fix-sm100-flashinfer-autotune-tests?quick_pull=1&title=%5BBugfix%5D%5BCI%2FBuild%5D%20Keep%20diffusion%20CPU%20unit%20tests%20host-independent%20on%20SM100%2B%20GPUs) |
| Related RFC | None. Small test-only bug fix, no RFC or issue needed. Related: #9998 (enabled FlashInfer autotune by default on SM100+), #10346 (same fix for the weight-sharing test). |
| Separate issue needed | No |
| Branch | `shijiew555:fix-sm100-flashinfer-autotune-tests` at `8bb727a72`, base `vllm-project/vllm-omni:main` |

The link opens the PR form with the title filled in.
Paste everything below the line into the description box.

<!-- PR BODY START -->
---

## Contributor intent (required human answers)

### 1. What outcome do you want to achieve?

The CPU-marked CI tests pass on developer machines with recent consumer GPUs (SM120), as they do in CI.

### 2. What is the highest-leverage variable for that outcome?

These CI tests are always needed, so they must give the same result on the consumer GPUs that local deployments use.

### 3. How will this work move that variable?

Light, test-only fixes. The tests set their config explicitly, or skip with a clear reason, instead of following defaults that depend on the host GPU. This makes later developers' work easier.

### 4. Whose customer or user needs does this address, and what are those needs?

Contributors and local users who run the test suite on recent consumer GPUs. They need a clean local run so that real failures stand out.

## Purpose

15 CPU-marked tests fail on a host with an SM100+ GPU (here an RTX 5080, SM120) and FlashInfer installed.
CI runs them on older GPUs, so the failures are not visible there.
The change is test-only.
Runtime behavior does not change.

Two host defaults cause the failures:

- 14 tests build a bare `DiffusionWorker` with `object.__new__` and call `execute_model`, `execute_model_batch` or `execute_stepwise`.
  Since #9998, FlashInfer autotune is on by default on CUDA SM100+.
  A real worker resolves the setting in `load_model()`.
  A bare worker resolves it lazily, enables it, and then fails with `AttributeError: 'DiffusionWorker' object has no attribute 'rank'` when it builds the per-rank cache path.
  The capability check reads NVML, so hiding the GPU with `CUDA_VISIBLE_DEVICES=` does not avoid it.
  On a full SM100 host the tests would also read and write the real autotune cache.
  The bare-worker helpers now set `_flashinfer_autotune_on = False`, which is what `load_model()` sets when autotune is off.
  #10346 fixed the same problem for the weight-sharing test.
- `test_backend_selection_reaches_the_dit_layer` (MammothModa2) selects `FLASH_ATTN` and checks that the DiT layer receives it.
  The CUDA platform rejects an explicit `FLASH_ATTN` on Blackwell without FA4, so the test depends on the host.
  The test now resolves backend names from the registry and skips the platform capability checks, so it checks only the config-to-layer routing.

A fix in the worker (for example, no autotune without `od_config`) would change product code only to support test doubles.
Real workers always have `od_config` and `rank`, so the tests are the right place for the fix.

The change also fixes two existing mypy errors in `tests/diffusion/test_diffusion_step_pipeline.py` (an override parameter name and an undeclared `device` attribute), because the local `mypy-3.10` commit hook fails on the touched file without them.

Whether SM120 consumer GPUs should auto-enable FlashInfer autotune is a separate product question for the #9998 owners.
This PR does not change it.

## Test Plan

**vLLM Version:** 0.33.0 (torch 2.14.0+cu130, FlashInfer 0.7.2.post1)

**vLLM-Omni Commit:** b576ed001 (upstream/main) + this change

GPU: NVIDIA GeForce RTX 5080 (SM120), driver 595.99.02.

Reproduce:

1. On an SM100+ host with FlashInfer installed, run:
   `CUDA_VISIBLE_DEVICES=0 python -m pytest -q -m "cpu and not gpu" tests/diffusion/diffusion_kv/test_worker_contract.py tests/diffusion/test_diffusion_step_pipeline.py tests/diffusion/models/mammoth_moda2/test_dit_attention.py tests/profile/test_profile_iteration_controls.py`
2. Run the same command with `CUDA_VISIBLE_DEVICES=` to check that nothing else changes.
3. `pre-commit run --files <changed files>`

## Test Result

- GPU visible (`CUDA_VISIBLE_DEVICES=0`), the 4 files above with `-m "cpu and not gpu"`: before `15 failed, 109 passed`; after `124 passed`.
- GPU hidden (`CUDA_VISIBLE_DEVICES=`), same files: before `24 failed, 100 passed`; after `9 failed, 115 passed`.
  The 14 bare-worker tests also fail before the change with the GPU hidden, because the capability check reads NVML.
  The 9 remaining failures are the same on both sides (other MammothModa2 attention tests, `RuntimeError: No CUDA GPUs are available` from a CUDA build of torch with no visible device).
  No test fails after the change that passed before.
- `pre-commit run --files <changed files>`: passed, including the local `mypy-3.10` hook.
- Not run: the full CPU suite, and an SM100 (B200) host.

AI assistance: Used Claude Code to draft the change, tests, and PR description. I reviewed the changes and ran the validation above.
