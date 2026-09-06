# Dirty-state limitation (irreducible)

Every frozen CCAO artifact consumed by this P0 pass was generated from a **dirty working
tree**. The generating commits and what survives of their working state:

| Provenance commit | Role | Tree dirty | Diff sha256 recorded | Diff **text** archived |
|---|---|---|---|---|
| `508dc1c2` | 994-tree baseline/config + v6 experiment spec | True | `2268bae6162a5164…` | False |
| `2aa0346a` | lower-rho extension experiment spec | True | `fa45001c231a63d8…` | False |
| `d3ef45f2` | final_local_results: rho=0 split audit, recalibration path, Delta_NL | True | **none** | False |

## Consequence

The recorded `git_diff_sha256` values let us *detect* that uncommitted changes existed, but
they do not let us *reconstruct* them. For `d3ef45f2` not even a diff hash was stored.

Therefore:

* Checking out a provenance commit in a worktree reproduces the **committed** state at that
  point, not the state that actually ran.
* If a frozen artifact fails to reproduce and the committed source is identical, the residual
  cannot be attributed further than **F-DIRTY — unreconstructable dirty-state uncertainty**.
* **F-DIRTY is a disclosure item, not a regeneration trigger** (plan §F.6). An R3 result
  classified F-DIRTY leaves the frozen artifacts standing, with the limitation stated.

## Status from this audit

* All provenance commits are ancestors of HEAD: **True**
* Executed-path source drift between HEAD and the provenance commits: **True**
* Provenance worktree required for the historical reproduction test: **True**
* Files with executed-path drift: ['run_temporal_cv.py', 'utils/delta_nl.py']
* Hashes re-verified: 19, mismatches: 1

This file is written before any reproduction result is interpreted, per plan §F.2b.
