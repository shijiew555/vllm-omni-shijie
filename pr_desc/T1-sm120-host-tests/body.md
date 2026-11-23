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

On a host with an SM120 GPU, 14 tests in the CPU-marked suite fail on main.
CI has no SM120 runner, so the failures are not visible upstream, but every contributor with an SM120 GPU hits them.
This PR changes only tests.
Product behavior is unchanged.

MiniMax-H3 CPU model tests (10 tests):

- The 9 tests in `TestMiniMaxH3Extractor` (TeaCache) and `test_transformer_baseline_zero_strength_and_request_isolation` (ControlNet) build a small `MiniMaxH3DiTModel` and run it on CPU.
- Since #7989, the model turns AdaLN weight offload on by default when the host GPU is SM120 and no `cache_config` overrides it.
- The test configs have no `cache_config`, so on an SM120 host the model enables offload, and the CPU forward raises `RuntimeError: MiniMax H3 AdaLN offload requires eager CUDA inference`.
- Fix: both test configs set `cache_config={"minimax_h3_adaln_offload": False}`.
- Trade-off: the product default looks at the host GPU, not at the device the model runs on. For serving this is correct, because a CUDA host runs the model on CUDA. Only a CPU-only test on a CUDA host sees the mismatch, so I fixed the tests and kept the default. If the H3 owners prefer the default to depend on the model's device, that can be a follow-up.

int8 and ConvRot CUDA smoke tests (4 tests):

- `TestGPUInt8Smoke::test_real_gpu_int8_apply_forward` and the 3 tests in `TestCudaConvRot` fail with `cutlass_scaled_mm ... Int8 not supported on SM120. Use FP8 quantization instead, or run on older arch (SM < 100)`.
- Cause: vLLM's CUTLASS int8 scaled_mm rejects int8 on SM100 and newer. `Int8LinearMethod` reaches it through vLLM's kernel selection: `CutlassInt8ScaledMMLinearKernel.is_supported` only checks for CUDA, so vLLM selects CUTLASS on SM100+ too (vLLM 0.33.0). ConvRot calls `ops.cutlass_scaled_mm` directly for its INT8 and W4A4 GEMMs.
- Fix: these tests skip when the CUDA device capability major is 10 or higher, with the reason `vLLM CUTLASS int8 scaled_mm does not support SM100+ GPUs.` The other int8 CUDA smoke tests (quantization and weight processing) still run on SM120 and pass.
- Out of scope: making production int8 and ConvRot fail early with a clear error on SM100+, and the vLLM kernel selection itself.

No user guide or design contract changes, because only tests change.

## Test Plan

Environment: RTX 5080 16 GB (SM120), driver 595.99.02, torch 2.14.0+cu130, vLLM 0.33.0, vLLM-Omni b576ed001 (main) and this branch.

Reproduction on main:

1. On an SM120 GPU, run `python -m pytest -q -n 4 -m "cpu and not gpu" tests/config tests/diffusion/quantization tests/diffusion/cache tests/diffusion/lora tests/diffusion/offloader tests/diffusion/models/minimax_h3`.
2. Observe 14 failures: 9 in `TestMiniMaxH3Extractor`, 1 in `test_minimax_h3_controlnet.py`, 1 in `TestGPUInt8Smoke`, 3 in `TestCudaConvRot`.

Checks:

- The command above, on main and on this branch.
- `python -m pytest -rs -q tests/diffusion/models/minimax_h3/test_minimax_h3_controlnet.py tests/diffusion/cache/test_teacache_extractors.py tests/diffusion/quantization/test_int8_config.py tests/diffusion/quantization/test_convrot_config.py` on this branch, with the GPU visible.
- The TeaCache, int8, and ConvRot files with `CUDA_VISIBLE_DEVICES=` (CUDA torch, no visible GPU), on main and on this branch.
- `pre-commit run --files` on the four changed files.
- To look for other MiniMax-H3 tests that build the model on CPU, the CPU-marked tests in `tests/diffusion/models/minimax_h3`, `tests/diffusion/models/taomate_h3`, `tests/diffusion/attention`, `tests/diffusion/model_loader`, and `tests/model_executor` on main. Only the ControlNet test above fails from this cause.

## Test Result

- Suite on main: `14 failed, 4090 passed, 14 skipped`.
- Suite on this branch: `4100 passed, 18 skipped` (the 4 new skips are the CUTLASS int8 tests).
- Changed files on this branch, GPU visible: `122 passed, 10 skipped`. The skip reasons show the 4 CUTLASS int8 tests; the other 6 are NPU-only tests.
- TeaCache, int8, and ConvRot files with no visible GPU: `5 failed, 58 passed, 6 skipped, 27 errors`, the same test IDs on main and on this branch. All are `RuntimeError: No CUDA GPUs are available`: the platform still reports CUDA when `CUDA_VISIBLE_DEVICES` hides every GPU. This is not caused by this PR and is out of scope. The new skip condition checks the device count, so it does not add a collection error in this setup.
- pre-commit: all hooks pass (ruff, mypy-3.10, CI marks, no new `torch.cuda` calls, SPDX).
- Not run: the CI CPU runner (no SM120 GPU there; the change keeps the same behavior on CPU-only and pre-SM100 hosts by construction), SM100 hardware.

AI assistance: Used Claude Code to draft the change, tests, and PR description. I reviewed the changes and ran the validation above.
