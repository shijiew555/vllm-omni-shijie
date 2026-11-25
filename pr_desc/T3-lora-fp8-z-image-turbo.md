# T3: [Model] Validate Z-Image-Turbo runtime LoRA with online FP8 quantization

| Field | Value |
| --- | --- |
| Opened | [#11126](https://github.com/vllm-project/vllm-omni/pull/11126), 2026-11-24 |
| Type | Pull request |
| Title | `[Model] Validate Z-Image-Turbo runtime LoRA with online FP8 quantization` |
| Open the PR | [Open PR in one click](https://github.com/vllm-project/vllm-omni/compare/main...shijiew555:vllm-omni-shijie:lora-fp8-z-image-turbo?quick_pull=1&title=%5BModel%5D%20Validate%20Z-Image-Turbo%20runtime%20LoRA%20with%20online%20FP8%20quantization) |
| Related RFC | [#1217 RFC: Continuous Diffusion Model Acceleration Support](https://github.com/vllm-project/vllm-omni/issues/1217), Feature x Feature table, cell LoRA Inference x FP8 Quant. Claim: [comment 6891432700](https://github.com/vllm-project/vllm-omni/issues/1217#issuecomment-6891432700) (2026-11-15, no reply yet). |
| Separate issue needed | No |
| Branch | `shijiew555:lora-fp8-z-image-turbo` at `f190ff50e`, base `vllm-project/vllm-omni:main` |

The link opens the PR form with the title filled in.
Paste everything below the line into the description box.

<!-- PR BODY START -->
---

Runtime PEFT LoRA with online FP8 quantization is marked unverified (❓) in the diffusion feature compatibility matrix.
This PR adds an exact-comparison CUDA test for Z-Image-Turbo with `quantization="fp8"` and documents the validated configuration.

Related to #1217; [task claim](https://github.com/vllm-project/vllm-omni/issues/1217#issuecomment-6891432700).
Complements #9996 (LoRA with module-level CPU offload on the same model). It does not touch the distill LoRA backend covered by #10162.

## Contributor intent (required human answers)

### 1. What outcome do you want to achieve?

Complete local, on-premise serving of open-source image, video and audio generation models for content creation on consumer GPUs, with vllm-omni as a more performant replacement for ComfyUI in these deployments.

### 2. What is the highest-leverage variable for that outcome?

Support for Z-Image-Turbo and more models in our local generation workflow, with vllm-omni as the backend.

### 3. How will this work move that variable?

Local generation uses many fine-tuned LoRA adapters, and FP8 is one of the popular precisions. This PR validates that runtime LoRA works with online FP8 quantization on Z-Image-Turbo.

### 4. Whose customer or user needs does this address, and what are those needs?

The users who run open-source image, video and audio models locally, on premise, for content creation, including teams that build agentic director or orchestrator tools on top of the serving layer. They need inference with fine-tuned LoRA adapters on FP8-quantized models.

## Purpose

- Runtime LoRA computes the base output through the layer's quantization method and adds the adapter delta in the pipeline dtype. So FP8 base weights and BF16 adapter weights should compose. The test checks this end to end. No production code changes.
- New test `tests/diffusion/lora/test_diffusion_lora_fp8.py::test_zimage_lora_online_fp8` runs Z-Image-Turbo with `quantization="fp8"` and two synthetic PEFT adapters:
- Adapter A: rank 8 on the fused `to_qkv` projection, the same adapter as #9996.
- Adapter B: rank 16 with the Diffusers per-projection names (`to_q`, `to_k`, `to_v`, `to_out.0`, `w1`, `w2`, `w3`) and `adaLN_modulation.0` on the first block. The manager must pack Q/K/V into the FP8 `to_qkv` and W1/W3 into the FP8 `w13`. `adaLN_modulation.0` stays unquantized in Z-Image, so B also covers LoRA on an unquantized layer inside the FP8 model. This is the module layout of common ai-toolkit Z-Image-Turbo adapters.
- Request sequence, run twice in one engine: base, A, B, A, A at scale 0, A, base, B. The first request runs before any adapter is loaded. Both adapters must change the image and must differ from each other. Every repeated state must match its first image pixel for pixel. Scale 0 and a request without an adapter must reproduce the base image exactly.
- The test is a separate file, not a change to `test_diffusion_lora.py`, because #9996 rewrites that file. This keeps the two PRs free of conflicts. The marks (`advanced_model`, `diffusion`, `hardware_test(res={"cuda": "L4"})`) match #9996, so the existing L4 merge job (`pytest tests/diffusion/lora/ -m 'advanced_model and cuda'`) collects it without Buildkite changes.
- Docs: a "Runtime LoRA with online FP8 quantization" section in the LoRA guide and the LoRA x FP8 cell of the compatibility matrix, scoped to Z-Image-Turbo. #9996 edits a different cell of the same matrix row, so the two PRs touch the same line; the rebase is trivial.

## Test Plan

- GPU: one RTX 5080 16 GB (SM120), driver 595.99.02, display server on the same GPU. Host RAM 30 GB.
- Python 3.12.3, torch 2.14.0+cu130, diffusers 0.40.0, transformers 5.19.0, flashinfer 0.7.2.post1.
- **vLLM Version:** 0.33.0
- **vLLM-Omni Commit:** 5638c045d (based on b576ed001)
- Model: local copy of `Tongyi-MAI/Z-Image-Turbo` (Diffusers layout), used through `MODEL_PREFIX`.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  python -m pytest -sv tests/diffusion/lora/test_diffusion_lora_fp8.py \
  -m 'advanced_model and cuda' --run-level advanced_model

CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/diffusion/lora -m "cpu and not gpu"

pre-commit run --files tests/diffusion/lora/test_diffusion_lora_fp8.py \
  docs/user_guide/diffusion/lora.md docs/user_guide/diffusion_features.md
```

On this 16 GB card, the first run without `expandable_segments` ran out of memory at startup while the transformer was quantized layer by layer (15.39 GiB in use, BF16 text encoder resident). With `expandable_segments:True` the test passes. The CI L4 has 24 GB.

FP8 with a trained adapter: [`tarn59/pixel_art_style_lora_z_image_turbo`](https://huggingface.co/tarn59/pixel_art_style_lora_z_image_turbo) (Apache-2.0, rank 32, all 30 blocks). Its ai-toolkit keys (`diffusion_model.*`) were renamed to `base_model.model.transformer.*` in a local PEFT directory; checkpoint key mapping is tracked in #8001. Settings: 1024x1024, 8 steps, guidance 0, seed 42, eager, TORCH_SDPA, prompts with the trigger `Pixel art style.`.

## Test Result

- `test_zimage_lora_online_fp8`: 1 passed in 36.50 s at 5638c045d (16 requests; an earlier run at 21391e099 also passed). All exact comparisons passed. Peak device memory during the measured requests: 12089 MiB (device-wide polling). Startup peak: about 15400 MiB (nvidia-smi, 5 s sampling). Median request latency at 256x256, 2 steps: 0.108 s, the same with and without an adapter.
- CPU LoRA tests (`tests/diffusion/lora -m "cpu and not gpu"`, GPU visible): 176 passed, 4 deselected in 9.23 s. With `CUDA_VISIBLE_DEVICES=` 24 of them fail with `No CUDA GPUs are available` (they need a visible CUDA device although marked CPU; unrelated to this change).
- pre-commit: all applicable hooks passed (ruff, mypy, test marks, SPDX, forbidden imports, CUDA-call policy, markdownlint).
- The CI merge-job selection collects the new test: `pytest --collect-only tests/diffusion/lora/ -m 'advanced_model and cuda'` lists it.

FP8 with the pixel-art adapter at 1024x1024, 8 steps (each adapter warmed up once, then 2 prompts measured):

| Adapter | Median latency (s) |
|---|---:|
| None | 2.83 |
| Pixel art (rank 32, 240 modules) | 3.09 |
| Synthetic adapter B (rank 16, 8 modules) | 2.85 |

Peak device memory 14409 MiB (device-wide polling), startup 31.6 s. The pixel-art style is clearly applied in the FP8 output, and the image without an adapter keeps the base style.

Not run:

- FP8+LoRA vs BF16+LoRA image comparison. BF16 Z-Image-Turbo does not fit this 16 GB GPU, and its module offload path needs more host RAM than this 30 GB machine has. The existing `fp8_z_image` case in `tests/diffusion/quantization/test_quantization_quality.py` covers FP8 vs BF16 quality without LoRA.
- The new test on an L4 or another 24 GB GPU.

AI assistance: Used Claude Code to draft the change, tests, and PR description. I reviewed the changes and ran the validation above.
