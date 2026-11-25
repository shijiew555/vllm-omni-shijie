# Rebase #8202 (Pi 0.5 R0) onto #8086 (F2)

| Field | Value |
| --- | --- |
| Status | Planned, not started |
| PR | [#8202](https://github.com/vllm-project/vllm-omni/pull/8202), branch `pi05_R0_optimization` |
| Requested by | chenchaoxu7575, 2026-11-20, after #8086 merged |
| Related | #6867 (tracker), #7709 (F2 design), #7836 (R1), #8331 (D0), #8821 (P0), #10128 (R2), #9941 (S0) |
| Visual map | Pi 0.5 Code Map artifact (private) |

## What #8086 changed

- Moved Pi 0.5 from `models/pi05/` to `models/pi/pi05/`. `models/pi05/` is now an alias for 4 submodules only.
- Moved shared mechanisms into `models/pi/common/`: prefix execution (`execute_prefix`), prefix embedding (`embed_multimodal_prefix`), eager attention and masks, dtype matching and the mixed-bf16 layout, the Euler step, checkpoint and pipeline helpers.
- Kept Pi 0.5-specific: AdaRMS suffix, `forward`, `denoise_step`, `sample_actions`.
- Changed mixed-bf16 rounding: the residual stream is cast to the decoder dtype before layer 0, in prefix and suffix.

## Plan

1. Port onto a fresh branch from upstream/main as about 5 logical commits, not a commit-by-commit rebase. Tag the old head as a backup.
2. Keep R0's code where it is: `Pi05KVCache`, the fused kernels and `_TILES` stay in `pi/pi05/modeling_pi05.py`; `cuda_graph_pi05.py` moves to `pi/pi05/`.
   Keep all public names, because D0, P0, R2 and S0 build on them.
3. Delete R0's copies of helpers that are now in `common`.
4. Eager prefix: call the shared `execute_prefix`, then write the K/V into `Pi05KVCache`. Keep one KV contract in `forward`.
5. Fused passes: cast the input to the decoder dtype and keep the residual buffer in that dtype, to match F2. The kernels already take their dtype as a parameter.
6. Rerun the fused, CUDA graph and eager parity tests against the new eager baseline, as chenchaoxu asked. fp32 should stay bit-identical to pre-rebase R0.
7. The user force-pushes (settings deny `git push --force`), then posts a summary on #8202 with the old-to-new map and a note to the 4 dependent PRs.

## Included already

- PR #1 on the fork (ZJLi2013, AMD gfx942 shared-memory stage cap) is merged into `pi05_R0_optimization` as `4585ec9c9`.
  On the RTX 5080 it changes nothing; tests pass and latency is within 0.05%.
  It still needs a gfx942 run by the contributor.
