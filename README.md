# agent-coord

Coordination and work records for contributions from this fork to `vllm-project/vllm-omni`.
This branch has no shared history with the code branches and is never merged into them.

## Layout

| Folder | Contents |
| --- | --- |
| `explorations/` | Findings and leads that may become PR work. Not yet approved. |
| `tasks/` | Tasks confirmed to become a PR: scope, branch, status, results, open items. |
| `agents/` | The subagent definitions used for this work (Claude Code format). |
| `pr_desc/` | Ready-to-post deliverables: one file per PR or issue, with the title, a one-click link, the related RFC, and the exact body. |

## Flow

1. A lead starts in `explorations/`.
2. When the user approves it, it gets a file in `tasks/` and a branch and worktree.
3. When the branch is reviewed, its post goes into `pr_desc/`.
4. Nothing is posted to GitHub until the user approves it.

## Rules

- In a `pr_desc/` file, the text after the `PR BODY START` marker is the exact text to post. Notes go in the header table or in `tasks/`.
- Never put private information (for example employer or startup details) in `pr_desc/`.
- Never use the em dash. Use a plain dash.
