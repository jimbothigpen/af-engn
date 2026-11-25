# Active Backports (`af-engn`)

Temporary upstream-PR backports live at the tip of `main` above permanent `af-engn` commits (`IQ*_K` / `IQ*_KT` / `IQ*_BN` / custom server features), prefixed with `backport(pr:<repo>#<num>): <summary>`.

## Upstream Sync Procedure

Follow this exact sequence whenever rebasing `main` onto a newer `upstream/master`:

1. Fetch upstream and tags:
   ```bash
   git fetch upstream --tags
   ```
2. Audit every row in the table below against `upstream/master` (via `gh pr view <num> --repo ggml-org/llama.cpp` and `git log upstream/master --grep="#<num>"`) to identify which upstream PRs have merged or been superseded.
3. Run an interactive rebase onto `upstream/master`:
   ```bash
   git rebase -i upstream/master
   ```
   - **Drop** every `backport(pr:...)` commit whose upstream PR has landed (never merge-conflict a stale backport against a newly merged upstream version).
   - **Drop** any local companion fix whose drop condition is satisfied (see table).
   - Replay only permanent `af-engn` commits and still-unmerged `backport(pr:...)` commits.
4. Prune dropped rows from this table, and check whether any remaining Unsloth backport needs a newer repin for the updated upstream base.
5. Verify read-only push protection remains intact on all donor remotes:
   ```bash
   git remote set-url --push upstream DISABLE
   git remote set-url --push unsloth DISABLE
   git remote set-url --push ik_llama DISABLE
   ```

## Active Backports Table

Base: `upstream` `b12635` (permanent `af-engn` tip `2dd3c1e21`).

| Local Commit | Source PR / SHA | Upstream Tracking PR (ggml-org) | Status / Drop Condition |
|---|---|---|---|
| `d22bd86fd` | `unslothai/llama.cpp#385` (`4aae13c13`) | Unsloth-only (Inkling-2 rel-bias row chunking) | OPEN upstream; drop when `inkling2.cpp` row-chunking lands in `upstream/master` |
| `f2c49639d` | `unslothai/llama.cpp#405` (`b5ed9f284`) | `ggml-org/llama.cpp#32218` (MERGED `e4b1cb531` + legacy fused layout `0a9105de4`) | MERGED on `upstream/master` (2026-11-24); drop on next `upstream/master` rebase (verify legacy fused drafter GGUFs if still in use) |
| `963befce7` | `unslothai/llama.cpp#397` (`805b04c23`) | Unsloth-only (keep DSpark fuse weights F32 in `llama-quantize`) | OPEN on Unsloth; drop once `llama-quantize` preserves DSpark fuse weights F32 in `upstream/master` |
| `79e6bd5a8` | `unslothai/llama.cpp#396` (`d96a8345a`, `9b6b8a4a0`, `bff724794`, `77941e048`) | Unsloth-only (looped layers device placement + shared host weight views) | OPEN on Unsloth; drop when looped-layer weight placement lands in `upstream/master` |
| `50ef06d25` | `unslothai/llama.cpp#421` (`975c44a44`, `b336a7e65`) | Unsloth-only (one MoE cache entry per expert weight, MCT2) | OPEN on Unsloth; drop when looped-layer MoE cache sharing lands in `upstream/master` |
| `05d2656d3` | `unslothai/llama.cpp#328` (`02428e283`) | Unsloth-only (Vulkan `top_k` misaligned input scratch + prealloc sync) | OPEN on Unsloth; drop when Vulkan misaligned `top_k` fix lands in `upstream/master` |
| `ff795f5ff` | `unslothai/llama.cpp#386` (`210fc2653` + `2c38e50c1` from `#378`) | Unsloth-only (banded FA + Inkling-2 graph folding; excludes Inkling-1 & `GGML_PREC_F32_PEDANTIC`) | OPEN on Unsloth; drop if upstream merges equivalent CUDA/CPU banded FA or supersedes with CUDA indexed-bias FA |
| `93bc1d0c9` | `unslothai/llama.cpp#413` (`627b5434a`, includes F16 cast fix over `260b1ffa4`) | Unsloth-only (CUDA fused relative-position bias in vector FA) | OPEN on Unsloth; drop when CUDA fused rel-bias vector FA lands in `upstream/master` |
| `8c5a7945e` | `ggml-org/llama.cpp#32740` (`239242322`) | `ggml-org/llama.cpp#32740` (MERGED 2026-11-23) | MERGED on `upstream/master`; drop on next `upstream/master` rebase |
| `bf984cf55` | `ggml-org/llama.cpp#32898` (`ce1e24697`) | `ggml-org/llama.cpp#32898` (OPEN) | OPEN upstream; drop once `#32898` merges into `upstream/master` |

## Companion Local Fixes (Non-`backport(pr:)` Commits)

| Local Commit | Purpose | Drop Condition |
|---|---|---|
| `ef33165cb` | `fix(cpu): enable indexed-bias FLASH_ATTN_EXT in CPU supports_op` — removes accidental early return in `ggml_backend_cpu_device_supports_op()` shipped in `#32740` so CPU reference & `test-backend-ops` run | Drop once `upstream/master` removes the early return in `ggml/src/ggml-cpu/ggml-cpu.cpp` |
| `7d5138f38` | `feat(inkling2): wire compact indexed-bias FLASH_ATTN_EXT for non-banded backends` — connects `inkling2.cpp` to `#32740`/`#32898` indexed-bias FA on Vulkan while keeping `#386` banded FA on CUDA and `#385` chunking fallback | Drop once `upstream/master` wires `inkling2.cpp` to compact indexed-bias `FLASH_ATTN_EXT` |

## Inspected & Not Ported

- `ggml-org/llama.cpp#32773` (OPEN): measurement report + optional `LLAMA_SPEC_ADAPT` / `p_min` patch in `common/speculative.cpp`. Its primary +68% RADV 16 GB decode fix is operational (`-ctkd q4_0 -ctvd q4_0`, already supported in `common/arg.cpp` + fixed `--spec-draft-n-max 4`); adaptive pausing regressed code throughput (80 -> 73 tok/s) and is unmerged. Re-evaluate only if merged upstream or requested.
