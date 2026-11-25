# Open leads

Evidence levels: **confirmed** (reproduced here), **hypothesis** (from code reading only).

| Lead | Evidence | Blocker or note |
| --- | --- | --- |
| LoRA x TeaCache and LoRA x Cache-DiT on Z-Image-Turbo (#1217 matrix) | Cells are unverified on main (confirmed). No overlapping PR found on 2026-11-15. | Needs a claim on #1217. Same weights and pattern as T3, so best done after T3. |
| MiniCPM-o 4.5, VoxCPM2 and Step-Audio2 CPU-marked tests fail on this SM120 host | Seen in the T1 search on `b576ed001` (confirmed failure, cause not diagnosed). | Check if the cause is the host GPU or only missing dev extras. |
| CPU-marked tests fail when torch has CUDA but no GPU is visible (GR00T, CosyVoice3, OmniVoice, merged LoRA, Fish KV cache, MammothModa2) | Confirmed with `CUDA_VISIBLE_DEVICES=`. | Probably a platform detection gap: the platform reports CUDA with zero visible devices. Low priority unless CI hits it. |
| Gepard #10227 fails under Model Runner V2 | Unclaimed on 2026-11-14 (confirmed). 1.1 GB weights. | Needs `nemo_toolkit[tts]==2.7.3` in a dedicated venv. |
| OpenPI decode: an `nd=0` numpy marker with empty data raises IndexError instead of a clear ValueError (`entrypoints/openpi/connection.py:88-110`) | Hypothesis. | Reproduce first. |
| Production int8 and ConvRot should fail early with a clear error on SM100+ | Follows from T1 (confirmed on vLLM 0.33). | Recheck after #10708 on vLLM 0.34. |
| vLLM `CutlassInt8ScaledMMLinearKernel.is_supported` checks only for CUDA | Confirmed on vLLM 0.33. | Upstream vLLM report, not vllm-omni. Recheck on vLLM 0.34. |
| FlashInfer autotune is on by default for SM120 consumer GPUs (#9998) | Confirmed. | Product question for the #9998 owners. |
| `Pi05AdaRMSNorm._norm` computes `x.float()` twice | From the #8086 review (chenchaoxu7575). | avraichur96 plans a separate cleanup. Do not take. |
