# PR and issue deliverables

Each file is one post: a header table, then the exact body after the `PR BODY START` marker.
The one-click link opens the GitHub "compare" page for the fork branch with the PR form open and the title filled in.
The body is not in the link, because the longest body would push the URL past GitHub's length limit.

| File | Title | Branch | Related RFC |
| --- | --- | --- | --- |
| `T1-sm120-host-tests.md` | `[Bugfix][CI] Keep MiniMax-H3 CPU and int8 CUDA tests independent of SM120 hosts` | `fix-sm120-host-tests` | none |
| `T2-flashinfer-autotune-tests.md` | `[Bugfix][CI/Build] Keep diffusion CPU unit tests host-independent on SM100+ GPUs` | `fix-sm100-flashinfer-autotune-tests` | none |
| `T3-lora-fp8-z-image-turbo.md` | `[Model] Validate Z-Image-Turbo runtime LoRA with online FP8 quantization` | `lora-fp8-z-image-turbo` | #1217 |

None of the three needs a separate issue.

## Before posting

The bodies report results measured on upstream/main `b576ed001` with vLLM 0.33.0.
Upstream main now needs vLLM 0.34 (#10365).
Run the checks again on the rebased branches, then update the results, SHAs and versions in each body.
See `tasks/` for what each task must check again.
