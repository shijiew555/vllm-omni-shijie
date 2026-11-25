# Explorations

Findings and leads that may become PR work.
A lead moves to `tasks/` only after the user approves it.

| File | Contents |
| --- | --- |
| `leads.md` | Open leads not yet turned into tasks, with evidence level and blockers |
| `2026-11-14-discovery-rounds.md` | Full report of discovery rounds 1 and 2: candidates, evidence, overlap checks, rejected items |

## Machine limits to check first

- One RTX 5080, 16 GB, SM120, shared with the display server. Use the GPU lock `/tmp/claude-1000/gpu.lock`.
- 30 GB host RAM. Run heavy jobs under `systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0`.
- Limit large downloads to 6 MB/s (`wget --limit-rate=6m`); at full speed they take GitHub and the Claude API offline.
- Upstream main needs vLLM 0.34 since #10365. The venv at `/home/sw999/vo/venvs/vllm033` is too old for current main.
