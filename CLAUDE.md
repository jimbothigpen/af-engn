IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## `af-engn` Fork: Upstream Sync & Backports

This repo is the private `af-engn` fork of `llama.cpp`. Keep `main` layered as `upstream/master` -> permanent `af-engn` commits -> temporary `backport(pr:...)` commits tracked in [BACKPORTS.md](BACKPORTS.md).

When syncing with `upstream/master`:
1. `git fetch upstream --tags`
2. Check every entry in [BACKPORTS.md](BACKPORTS.md) against `upstream/master` (`gh pr view` / `git log upstream/master`) to see which upstream PRs have merged or been superseded.
3. Run `git rebase -i upstream/master` and **drop** every `backport(pr:...)` commit (and companion local fix) whose upstream PR has landed — never resolve merge conflicts on a stale backport against a merged upstream version.
4. Replay only permanent `af-engn` commits + still-unmerged `backport(pr:...)` commits, prune dropped rows from [BACKPORTS.md](BACKPORTS.md), and check whether remaining Unsloth backports need repinning.
5. Keep `upstream`, `unsloth`, and `ik_llama` push URLs set to `DISABLE`.

