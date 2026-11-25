# PR candidates - discovery round 1

Written by the explorer controller, 2026-11-14 22:50 PST.
Base for all code reading: `upstream/main` e937f1c9e (2026-11-15).

## Summary

Discovery lost about 3 hours to a network outage (a Hugging Face download saturated the link; api.github.com and the Claude API both timed out, and explorer E1 died with an API error and had to be resumed).
Only one explorer ran (E1: test-suite bug hunt).
Result: one confirmed small bugfix (candidate 1, reproduced here, no claim needed), one roadmap validation task that needs a claim and a 32 GB download (candidate 2), and a weak follow-on (candidate 3).
Recommendation: approve candidate 1 now; approve candidate 2 only if the user accepts the claim wait and the throttled download (~90 min) before work can start.

Important environment facts for workers:
- The repo venv (`.venv`, vllm 0.30.0) cannot import upstream/main (needs vllm 0.33.0 since #9648): `AttributeError: module 'vllm.v1.attention.backends.utils' has no attribute 'compute_mm_prefix_ranges'` at `vllm_omni/utils/mm_prefix.py:25` (confirmed by me and E1).
  A working scratch venv exists: `/tmp/claude-1000/pr-run/explorer-testhunt/venv` (Python 3.12, vllm 0.33.0, torch 2.14.0+cu130), used with `PYTHONPATH=<upstream/main archive>`. Editable install of an archive fails (setuptools-scm, no git); a real worktree on upstream/main would install fine.
- Large HF downloads at full speed knock GitHub and the Claude API offline on this link. `wget --limit-rate=6m` kept both responsive (GitHub ~0.35 s, measured). Script: `/tmp/claude-1000/pr-run/explorer-controller/dl_zimage.sh` (throttled, resumable with `-c`, downloads into `/home/sw999/models/Z-Image-Turbo`). Remaining ~32 GB takes ~90 min at 6 MB/s.

---

## Candidate 1 (recommended, no claim needed)

**Title:** `[Bugfix][CI] Keep MiniMax-H3 TeaCache and int8 CUDA tests independent of SM120 hosts`

**Intent:** CPU-marked MiniMax-H3 TeaCache extractor tests and the int8/ConvRot CUDA smoke tests fail on any RTX 50xx (SM120) host; make them pass there without changing product behavior.

**Evidence (confirmed, reproduced on this RTX 5080 by E1, upstream/main e937f1c9e, vllm 0.33.0, torch 2.14.0+cu130):**
- Command: `flock /tmp/claude-1000/gpu.lock python -m pytest -q -p no:cacheprovider -n 4 -m "cpu and not gpu" tests/config tests/diffusion/quantization tests/diffusion/cache tests/diffusion/lora tests/diffusion/offloader` (with `PYTHONPATH=/tmp/claude-1000/pr-run/explorer-testhunt/base`, scratch venv) -> `13 failed, 2692 passed, 12 skipped` in 56 s.
- A (9 tests): all of `tests/diffusion/cache/test_teacache_extractors.py::TestMiniMaxH3Extractor::*` fail with `RuntimeError: MiniMax H3 AdaLN offload requires eager CUDA inference` (`vllm_omni/diffusion/models/minimax_h3/minimax_h3_blocks.py:700`).
  Root cause: `vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py:818-829` turns AdaLN weight offload on by default when the *host* GPU is SM120 (`is_cuda()` and capability `(12, 0)`), introduced by #7989. The CPU test fixture `_minimax_h3_small_od_config` (`tests/diffusion/cache/test_teacache_extractors.py:854-876`) has no `cache_config`, so on an SM120 host the model enables offload and the CPU forward raises. I verified the code and fixture by reading.
- B (4 tests): `tests/diffusion/quantization/test_int8_config.py::TestCudaInt8Smoke::test_real_cuda_int8_apply_forward` and 3 tests in `tests/diffusion/quantization/test_convrot_config.py::TestCudaConvRot` fail inside vLLM: `cutlass_scaled_mm ... Int8 not supported on SM120. Use FP8 quantization instead, or run on older arch (SM < 100)` (path via `vllm_omni/quantization/int8_config.py:493`). Tests are gated only by `@cuda_available` (`test_int8_config.py:612`, `test_convrot_config.py:130`).
- CI has no SM120 runner, so these failures are invisible upstream; any contributor with an RTX 50xx card hits them.
- Side result: #10115 (HunyuanImage3 revision test "Mock object is not subscriptable") does not reproduce (5 runs, 7/7 pass); fixed on main by #9850 which sets `config.diffusion_attention_schedule = None`. The PR description can mention it, or the user can comment on #10115.

**Overlap check:** `gh pr list --search` "minimax_h3_adaln_offload", "TestMiniMaxH3Extractor", "int8 SM120 test skip": no open PR on this. Related merged/closed: #8125 (H3 TeaCache RoPE init), #5997 (FakeBackend), #7989 (introduced the default). Open #10243 edits `minimax_h3_transformer.py` but not the offload default or this test (checked its diff). No overlap with #8202 or #8086.

**Proposed scope:**
- A: add `cache_config={"minimax_h3_adaln_offload": False}` to `_minimax_h3_small_od_config` (one line plus a comment).
- B: skip the int8 CUTLASS smoke tests when device capability major >= 10 (vLLM's CUTLASS int8 path rejects SM100+/SM120), with a reason string naming the vLLM error.
- Out of scope: changing the product default to depend on the model's device instead of the host GPU (mention in the PR as a follow-up question for H3 owners); making production int8 fail fast with a clearer error on SM100+ (separate question, maybe a later PR).

**Validation plan:** on this RTX 5080: `tests/diffusion/cache/test_teacache_extractors.py` and `tests/diffusion/quantization` before (13 failures) and after (0 failures), under the GPU lock, < 2 min; same tests with `CUDA_VISIBLE_DEVICES=` to show nothing else changes; `pre-commit run --files`. Must run in a venv with vllm 0.33.0 (scratch venv above, or a fresh venv in a worktree on upstream/main).

**Why the project benefits:** SM120 consumer cards are an explicit target (#10007/#10008, Q4 roadmap SM120 items); the CPU suite should be green on them.

**Claim status:** none needed (test-only bugfix, no issue). Optionally open a short issue first; not required.

**Risks / effort:** XS-S (~10-20 LOC). Risk: reviewers may prefer the product-side fix for A; the PR should state the trade-off. Could be split into two PRs if reviewers want; recommend one PR titled for both since both are "SM120 host breaks tests".

---

## Candidate 2 (needs claim, depends on a download)

**Title:** `[Test][LoRA] Validate runtime PEFT LoRA with online FP8 quantization on Z-Image-Turbo`

**Intent:** close the unverified "LoRA Inference x FP8 Quant" cell of the diffusion feature matrix for one model, with an exact-comparison CUDA test and docs, fixing any bug found on the way.

**Evidence**
- Roadmap: #1217 "Feature x Feature" table has the whole `LoRA Inference` row as help-wanted (confirmed, issue body). On upstream/main, `docs/user_guide/diffusion_features.md:271` shows every LoRA cell as `❓` (confirmed).
- Precedent with the exact same shape, open and not yet merged: #9996 "[Model] Validate Z-Image-Turbo LoRA with module-level CPU offload" (192+/115-, touches `tests/diffusion/lora/test_diffusion_lora.py`, `docs/user_guide/diffusion/lora.md`, `docs/user_guide/diffusion_features.md`), claimed in #1217 by zzznnn-1 on 2026-11-07 (confirmed).
- Code path (confirmed by reading): runtime LoRA computes the base output through `self.base_layer.quant_method.apply(...)` (`vllm_omni/lora/layers/base_linear.py:121`) and adds the delta with plain matmuls; LoRA buffers use the pipeline dtype (`vllm_omni/lora/manager.py:811`, `lora_dtype=self.dtype`), so FP8 base weights should not leak into the adapter buffers. Hypothesis: it works; the risk points are fused QKV slices on `QKVParallelLinear` under `Fp8LinearMethod`, adapters on layers in FP8 `ignored_layers`, and adapter switching after FlashInfer autotune (fp8.md says the batch key includes LoRA).
- Related but different: #10162 (open) rejects the *distill* LoRA backend on quantized weights at startup; it does not cover runtime PEFT LoRA. The new test should assert the PEFT path is not caught by that check.
- Feasibility on this machine (confirmed by third-party data, not yet run here): Z-Image-Turbo online FP8 runs on a 16 GB SM120 card (RTX 5060 Ti): FP8 no offload 12.98/15.49 GiB alloc/reserved, ~9.3 s per 1024x1024 request; FP8 + CPU offload 8.51/10.78 GiB (#10008, comment by mocias, 2026-11-10). This machine: RTX 5080 16 GB, 30 GB host RAM. Weights are public (not gated), 33 GB (transformer is stored fp32: 24.6 GB). About 0.9 GB downloaded so far in `/home/sw999/models/Z-Image-Turbo`, plus a partial HF cache copy.

**Overlap check:** `gh pr list --search` for "lora fp8", "lora quantization", "lora compatibility", "lora offload", "lora teacache", "lora cache-dit": only #9996 (offload), #10162 (distill LoRA reject), #6994, #6544 (other LoRA bugs) - none covers PEFT LoRA x FP8. #1217 comments: only LoRA claim is zzznnn-1 (module offload).

**Proposed scope**
- Extend `tests/diffusion/lora/test_diffusion_lora.py` (rebase on or coordinate with #9996, which restructures the same test) with an FP8 case: synthetic rank-8/16 PEFT adapters on the fused QKV, base vs adapter A vs B vs A-again vs scale 0, all under `quantization="fp8"`; compare FP8+LoRA outputs to FP8 base (adapter must change output, A must be reproducible bit-exact, scale 0 must restore base) and report PSNR/SSIM of FP8+LoRA vs BF16+LoRA.
- Docs: one row in `docs/user_guide/diffusion/lora.md` and the `LoRA x FP8 Quant` cell in `docs/user_guide/diffusion_features.md`, scoped to Z-Image-Turbo.
- Out of scope: other quant methods, distill LoRA, multi-GPU, other models.

**Validation plan:** the new CUDA test on the RTX 5080 (FP8, with CPU offload if needed to fit 16 GB), about 5-10 min per run after weights are local; existing CPU LoRA tests; `pre-commit run --files`.
Expected run time for the whole task: download 25-90 min (depends on throttle), then 2-3 h implementation and validation.

**Why the project benefits:** FP8 is the main way consumer GPUs fit these models, and users combining it with adapters currently get no statement of correctness.

**Claim status:** needs claim (comment under #1217, owner @wtomin). Draft:

> Hi @wtomin, I'd like to take a narrowly scoped Feature x Feature validation task from #1217: runtime PEFT LoRA x online FP8 quantization for `Tongyi-MAI/Z-Image-Turbo`.
> The LoRA row of the compatibility matrix is still unverified. I plan to check adapter activation, A-to-B-to-A switching and zero scale under FP8 with exact output comparisons, compare FP8+LoRA against BF16+LoRA, and contribute a CUDA test plus model-scoped docs, fixing anything that breaks. This complements #9996 (LoRA x module-wise CPU offload) and does not touch the distilled-LoRA path covered by #10162.
> Hardware: one RTX 5080 16 GB (SM120), Linux.
> If this is already covered, please let me know.

**Risks / effort:** effort S-M (~150-250 LOC incl. docs). Risk: conflict with #9996 in the same test file (coordinate or rebase); FP8 may need CPU offload to fit with the BF16 text encoder (Z-Image text-encoder FP8 from #1338 is an alternative); the claim may not be answered quickly.

---

## Candidate 3 (optional, weak, same weights, needs claim)

**Title:** `[Test][LoRA] Validate runtime PEFT LoRA with TeaCache and Cache-DiT on Z-Image-Turbo`

**Intent:** same pattern as candidate 2 for the `LoRA x TeaCache` and `LoRA x Cache-DiT` cells (Z-Image supports both, #1217 table: TeaCache #817, Cache-DiT done).
**Evidence:** matrix cells `❓` (`docs/user_guide/diffusion_features.md:271`, confirmed); no overlapping PR in the searches above (confirmed). Whether caching reuses residuals correctly across adapter switches is a hypothesis to test.
**Scope / validation:** as candidate 2, cache backend instead of FP8; should only be done after candidate 2 lands or by the same worker, to avoid two PRs editing the same test.
**Claim:** needs claim; can be added as a second sentence to the candidate 2 claim.
**Risks / effort:** S-M; lower priority than candidate 2; weak if run as a separate PR in parallel.

---

## Unverified leads (not recommended yet)

- GR00T CPU-marked tests (`tests/diffusion/models/gr00t/test_noise_seed.py`, `test_pipeline.py::test_forward_returns_dict_actions_in_output`) fail with `torch.AcceleratorError: no CUDA-capable device` when torch has CUDA but no visible GPU (`pipeline_gr00t.py:169`, device from `get_local_device()`). Environment quirk, low value (E1, confirmed reproduction).
- OpenPI edge cases (E1, hypothesis): `vllm_omni/entrypoints/openpi/connection.py:88-110` an `nd=0` numpy marker with empty `data` raises IndexError instead of a clean ValueError; `serving.py:276` `obs.pop("seed")` mutates the caller dict (harmless today).

## Checked and rejected

- Pi0.5 roadmap #6867: every item has an owner (F2 #8086, S0 #9941/#9237, R0 = user's #8202, P0 #8821, R1 avraichur96, R2 Polarisyjr #10128, D0 #8331). Note: #8202 has three inline review comments the tracker owner asked the user to address (2026-11-03 status update).
- VLA/OpenPI CPU tests on upstream/main: 225 passed, 3 skipped, 0 failed (E1), so no bugfix there.
- GR00T compile #6448: claimed by HaoyangChen23. `GR00T_NOISE_SEED` row in #6232: already removed by #7253 (tracker is stale).
- LingBot-VLA 2.0 #8812: model over 16 GB (16.5 GiB VLM + 6.7 GiB action expert, FP32).
- #3186 kernel matrix: every row claimed or closed negative.
- #1854 text-encoder FP8 rows: nearly all taken by 0z5a (#7819-#7873); remaining models (Qwen-Image, FLUX.1-dev, Wan, ...) do not fit 16 GB. F5-TTS FP8 row: model not in the repo.
- #10008 consumer 8-16 GB tiers: crowded (16 GB SM120 Z-Image/Wan already reported by mocias).
- #10115: does not reproduce, already fixed on main by #9850 (see candidate 1).
- Open bug issues 10370, 10284, 9638, 9709, 9671, 9886: all have owners or open PRs.
- #7093 chunk-decode CUDA graphs: large framework RFC, not small.
- #6232 env-switch packages: claimed (Cosmos3, MiniCPM-o, InternVLA-A1, MiMo, OmniVoice, CosyVoice3 asked today); remaining ones are for models that need checking for fit; not explored.

---

## Round 2 early note (23:05, explorer controller) - relevant to T1 / candidate 1

A third SM120-host CPU-test failure of the same kind, found on upstream/main e937f1c9e with the scratch venv and the GPU visible:
`tests/profile/test_profile_iteration_controls.py::test_diffusion_execution_advances_native_iteration_schedule[execute_model|execute_model_batch|execute_stepwise]` -> `AttributeError: 'DiffusionWorker' object has no attribute 'rank'` at `vllm_omni/diffusion/worker/diffusion_worker.py:704`.
Cause: the test builds `DiffusionWorker` with `object.__new__` and no `od_config`; `_flashinfer_autotune_wanted(None)` (`diffusion_worker.py:346`) auto-enables FlashInfer autotune on CUDA capability >= 10.0 (SM120 counts) when FlashInfer is installed, then `_flashinfer_autotune_cache_path(od_config, self.rank)` reads the missing `rank`.
Passes with `CUDA_VISIBLE_DEVICES=` (not CUDA). Test-side fix like candidate 1 A (e.g. set `worker._flashinfer_autotune_on = False` or an `od_config` with `diffusion_flashinfer_autotune=False`). The T1 worker may want to fold it in.

---

## Round 2

Written by the explorer controller (restart run), 2026-11-14, finished before 00:10 PST.
Base: `upstream/main` e937f1c9e archive at `/tmp/claude-1000/pr-run/explorer-testhunt/base`, scratch venv `/tmp/claude-1000/pr-run/explorer-testhunt/venv` (vllm 0.33.0, torch 2.14.0+cu130). No explorers were started (slot held by the T1 worker).

### What was run

- CPU suites not covered in round 1, `-m "cpu and not gpu"`, `-n 6`:
  - Run 1 (`CUDA_VISIBLE_DEVICES=`): tests/entrypoints, model_executor, engine, worker, worker_v2, core, inputs, utils, metrics, platforms, protocol, reasoning, quantization, kernels, attention, distributed, dfx, host_weight_runtime, model_extras, watermarking, tools, profile, rl_rollout, clients -> `42 failed, 20478 passed, 198 skipped, 15 errors` in 294 s. Log: `/tmp/claude-1000/pr-run/explorer-controller/r2_cpu1.txt`.
  - Run 2 (GPU visible, under the GPU lock): every tests/diffusion subfolder and file except quantization/cache/lora/offloader, plus tests/model_executor/models and tests/entrypoints -> `34 failed, 23199 passed, 133 skipped, 4 errors` in 389 s. Log: `r2_cpu2.txt`.
  - Re-ran every failing file with the GPU visible: `r2_gpu_rerun.txt`.
- Triage of failures:
  - Environment only (not bugs): missing dev extras in the scratch venv (`hyperpyyaml` via `step-audio2`, `torchmetrics`, `openpyxl`, `voxcpm`, `hymm`, `cosyvoice2`, `opentelemetry.instrumentation`); and "No CUDA GPUs are available" for CPU-marked tests when torch has CUDA but the device is hidden (CosyVoice3 components, OmniVoice fused load, merged LoRA, Fish KV cache; all pass with the GPU visible).
  - Real: 16 tests fail only because the host GPU is SM120 (see R2-A and the T1 note below).
- TTS sweep (user guidance): #1217 has no TTS rows; #1854 TTS rows are Voxtral (done) and F5-TTS (model not in repo). TTS trackers #2115, #3535, #4676, #4176, #9387 (MRV2 per model) are owner-driven and every listed item is claimed. Weekly CUDA build 19353 forced Model Runner V2 on all TTS lanes (#10073, `VLLM_OMNI_CI_MODEL_RUNNER=v2`), which produced a family of TTS failures (#10224 Breeze - claimed by searlion; #10225 CosyVoice3; #10227 Gepard; #10228 GLM-TTS; #10232 IndexTTS2 - #8787; #10233 Ming - wuli666; #10239 Voxtral - #10244/#9599). Maintainer comment on each: "adapt to v2, do not downgrade back to v1".

### Result

One new candidate meets the bar (R2-A).
The second slot stays empty: I did not lower the bar.
Plus two additions for T1 (same root causes as candidate 1, listed below).

### Additions for T1 / candidate 1 (same root cause, give to the T1 worker)

- `tests/diffusion/models/minimax_h3/test_minimax_h3_controlnet.py::test_transformer_baseline_zero_strength_and_request_isolation` -> `RuntimeError: MiniMax H3 AdaLN offload requires eager CUDA inference`. Same SM120 host default as candidate 1 A (confirmed, GPU visible; passes the AdaLN check with CUDA hidden).
- The profile test in the early note above belongs with R2-A (FlashInfer autotune root cause), not T1.

### Candidate R2-A (recommended, no claim needed)

**Title:** `[Bugfix][CI] Keep DiffusionWorker unit tests off FlashInfer autotune on SM100+ hosts`

**Intent:** 15 CPU-marked diffusion tests fail on any SM100+/SM120 host with FlashInfer installed; make them independent of the host GPU without changing runtime behavior.

**Evidence (confirmed, this RTX 5080, upstream/main e937f1c9e, scratch venv):**
- Command: `CUDA_VISIBLE_DEVICES=0 flock /tmp/claude-1000/gpu.lock python -m pytest -q -p no:cacheprovider -m "cpu and not gpu" tests/diffusion/diffusion_kv/test_worker_contract.py tests/diffusion/test_diffusion_step_pipeline.py tests/diffusion/models/mammoth_moda2/test_dit_attention.py tests/diffusion/models/minimax_h3/test_minimax_h3_controlnet.py tests/profile/test_profile_iteration_controls.py` -> `16 failed, 144 passed` (1 of these 16 is the H3 controlnet test, which goes to T1).
- 14 tests, `AttributeError: 'DiffusionWorker' object has no attribute 'rank'` at `vllm_omni/diffusion/worker/diffusion_worker.py:704`:
  - `tests/diffusion/test_diffusion_step_pipeline.py::TestWorker::*` (9), `tests/diffusion/diffusion_kv/test_worker_contract.py` (2), `tests/profile/test_profile_iteration_controls.py::test_diffusion_execution_advances_native_iteration_schedule[*]` (3).
  - Cause: these tests build `object.__new__(DiffusionWorker)` without `od_config`/`rank`. Since #9998 (merged 2026-11-07) `_flashinfer_autotune_wanted(None)` (`diffusion_worker.py:346`) auto-enables autotune on CUDA capability >= 10.0 with FlashInfer installed. Checked here: `wanted(None)= True`, `has_flashinfer True`. The worker then calls `_flashinfer_autotune_cache_path(od_config, self.rank)`. On a real SM100+ host the tests would also touch the per-rank autotune cache file.
  - CI runs these on non-SM100 hosts, so the failure is invisible upstream. Precedent: #10346 (merged 2026-11-14) fixed the same seam for the weight-sharing test with `_flashinfer_autotune_wanted=mocker.Mock(return_value=False)`; `tests/diffusion/test_diffusion_worker_flashinfer_autotune.py` sets `od_config`/`rank` explicitly.
- 1 test, `tests/diffusion/models/mammoth_moda2/test_dit_attention.py::test_backend_selection_reaches_the_dit_layer`: `ValueError: FLASH_ATTN was explicitly selected but is unsupported (Blackwell requires CuTe FlashAttention-4 ...)` from `vllm_omni/platforms/cuda/platform.py:258`. The test asks for `FLASH_ATTN` and so depends on the host GPU. Fix: patch the platform's backend resolution in the test, or pick a backend that is valid everywhere for the routing check.

**Overlap check:** `gh pr list --search` "flashinfer autotune", "SM120 tests OR blackwell host test", issue search "has no attribute 'rank'": only #9998 (introduced the default), #10346 (fixed the weight-sharing test only), #8181 / #8414 (unrelated SM120 kernels). No open PR touches these test files for this. No overlap with T1, which covers the H3 offload and int8 CUTLASS tests.

**Proposed scope:** test-only.
- Disable autotune in the bare-worker helpers of the 3 files (`worker._flashinfer_autotune_on = False`, or patch `_flashinfer_autotune_wanted` to return False as #10346 does). Optionally add a shared helper, since about 10 test files build bare `DiffusionWorker`s.
- Make the MammothModa2 backend-routing test independent of the platform.
- Out of scope: whether SM120 consumer cards should auto-enable FlashInfer autotune. This is a product question for the #9998 author (xuechendi); mention it in the PR.

**Validation plan:** same command before (15 failures from this scope) and after (0), under the GPU lock, < 1 min; same files with `CUDA_VISIBLE_DEVICES=` to show nothing else changes; `pre-commit run --files`. Needs a vllm 0.33.0 venv (scratch venv or a worktree install).

**Why the project benefits:** SM120/SM100 developers get a green CPU suite. Same reasoning as candidate 1, with a merged precedent for this exact seam.

**Claim status:** none needed (test-only bugfix).

**Risks / effort:** XS (~15-30 LOC, 3-4 test files), about 1 h including validation. Alternative: fold it into T1 if T1 is not pushed yet. It is a different root cause from T1, so a separate PR is clean and keeps T1 small.

### Checked and rejected in round 2

- #10228 GLM-TTS weekly voice-clone 500 under forced MRV2: does not reproduce at e937f1c9e (hypothesis-level: V2 selection came from `VLLM_OMNI_CI_MODEL_RUNNER=v2` via the test helper, but the runner choice was not independently confirmed in the stage logs).
  Weights downloaded to `/home/sw999/models/GLM-TTS` (+ `/home/sw999/models/campplus/campplus.onnx`).
  `test_voice_clone_zh_sync` on the RTX 5080: server starts, `POST /v1/audio/speech HTTP/1.1" 200 OK`. The test then fails only locally because `ffmpeg` is missing for the Whisper transcript check (`glm_v2_sync.log`).
  The voice-clone frontend (WhisperVQ tokens 137, CampPlus 192-d, mel 273x80) also works on CPU.
  Likely fixed on main by #10244 (generic chunk/postprocess MRV2 fixes, merged after build 19353). At most a comment on #10228 after a full run with ffmpeg installed. Script: `/tmp/claude-1000/pr-run/explorer-controller/scripts/run_glm_e2e.sh <v1|v2> <-k> <log>`.
- #10305 Qwen3-TTS cached-predictor ASR flake: open PR #10358.
- #9717 MOSS-VoiceGenerator `revision` kwarg: draft PR #10377; root cause is in the remote processor/Transformers.
- #10225 CosyVoice3 offline engine init under MRV2: codec MRV2 is covered by #9294 (#9387 thread).
- #10227 Gepard SSE under MRV2: unclaimed, 1.1 GB weights, but needs `nemo_toolkit[tts]==2.7.3` with overrides. Installing it into the shared scratch venv would risk the T1 worker's environment. Possible later with a dedicated venv.
- #10224 Breeze (claimed by searlion), #10233 Ming (wuli666), #10239 Voxtral (#9599/#10244), #10232 IndexTTS2 (#8787).
- TTS roadmaps #2115/#3535/#4676/#4176/#9387: all items owned. #1217: no TTS rows. #1854: F5-TTS not in repo.
- Good-first-issue items #4007 (PR #6266), #5522 (PR #5555), #4226 (PR #4229), #8555 (PR #8689), #4256 (assigned): taken.
- Remaining CPU-suite failures: environment only (missing dev extras in the scratch venv; CUDA-built torch with a hidden device). The dev extras are declared in `pyproject.toml`, so this is not a packaging bug.

### Environment notes

- Downloads: GLM-TTS 8.9 GB done at 4 MB/s. Z-Image-Turbo download still running (controller's job), at about 5.1 GB of 33 GB at 23:02.
- Mistake to avoid: `pkill -f dl_glmtts.sh` from a Bash call killed that call's own shell (exit 144) and hung the tool for about 40 min. Kill by PID instead.
