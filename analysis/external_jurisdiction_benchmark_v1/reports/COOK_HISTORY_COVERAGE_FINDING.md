# Correctness finding: Cook's (and Allegheny's) Assessor History delivery is truncated to 2019

Per instruction: "if baseline construction reveals a correctness issue, stop and report it; do not
redefine the residential cohort or temporal design." Reporting now. Cohort membership and
`temporal_design.yaml` were NOT changed in response to this.

## What happened

Cook's baseline CV search (`baseline/cook_baseline_config.json`) used only **4 of the frozen 7
expanding calendar-year folds** (2021-2024 validated; 2018/2019/2020 were skipped with `n_train=0`
or too small). Wayne and Philadelphia used all 7 folds without issue.

## Root cause, verified against the raw Dewey delivery (not a caching artifact)

```
allegheny   min ASSESSORHISTORYYEAR across sampled shards: 2019
cook        min ASSESSORHISTORYYEAR across sampled shards: 2019
maricopa    min ASSESSORHISTORYYEAR across sampled shards: 2015
king        min ASSESSORHISTORYYEAR across sampled shards: 2015
miami_dade  min ASSESSORHISTORYYEAR across sampled shards: 2006
middlesex   min ASSESSORHISTORYYEAR across sampled shards: 2006
```

Checked directly against `data/dewey-downloads/cookcounty-2016-2025-all-features/*.parquet` and the
equivalent Allegheny folder -- **both actually begin in 2019**, despite the "2016-2025" folder name.
This is a property of the raw source delivery, not of `build_county_caches.py` or
`build_modeling_tables.py`: our cache is a faithful predicate-pushdown copy of what Dewey shipped.

Since History match requires the latest record with `assessed_through` (Dec 31 of
`ASSESSORHISTORYYEAR`) **strictly before** the sale, and Cook/Allegheny's earliest possible
`assessed_through` is 2019-12-31, **no sale before 2020-01-01 can have a valid safe-history match in
either county, by construction.** Cook's raw Recorder eligible sales in 2016-2019 are plentiful
(166,720 / 210,247 / 187,651 / 165,729 in 2016-2019) -- they are not absent, they are structurally
unmatchable to any strictly-prior assessment record.

Maricopa, King, Miami-Dade, and Middlesex do not have this problem: their History coverage starts
at or before their labeled range with adequate buffer (King's 2015 start gives one year of buffer
before the 2016 sale window, the same buffer convention v3 used for St. Louis).

## What this means for the frozen design

`temporal_design.yaml`'s seven expanding calendar-year folds (2018-2024) assumed every jurisdiction
has usable History coverage from 2016 onward. That assumption holds for Wayne, Philadelphia,
St. Louis, Maricopa, King, Miami-Dade, and Middlesex, but **not for Cook or Allegheny**, whose
effective development window is 2020-2024 (5 years, 4 validation folds) rather than 2016-2024
(9 years, 7 validation folds).

Cook's baseline CV numbers already produced (`grid_07`, CV mean R2_price 0.681, RMSE_log 0.281) are
computed correctly given the data actually available -- they are not wrong, only over a shorter,
more recent window than Wayne/Philadelphia's. This affects cross-jurisdiction comparability, which
is a stated goal of this benchmark, so it is reported rather than absorbed silently.

## Decision needed (not made here)

This requires a human decision because every option trades off differently against the
cross-jurisdiction comparability goal:

1. Accept Cook/Allegheny's shorter effective window and document it per-jurisdiction (what already
   happened mechanically here, via the existing small-fold skip guard) -- results stand as computed,
   annotated with fewer folds and a later development start.
2. Define development-window start per jurisdiction as
   `max(2016, earliest available History year + 1-year buffer)` -- consistent with the buffer
   convention already used for St. Louis in v3, but this IS a change to the frozen temporal design
   and requires explicit authorization, applied uniformly (not just to fix Cook).
3. Exclude Cook and/or Allegheny from the primary cross-jurisdiction comparison, keeping them as a
   separately labeled sensitivity given their shorter history depth.
4. Some other resolution the user specifies.

No cohort or temporal-design change has been made. Wayne and Philadelphia's 7-fold results are
final for Step 5; Cook's 4-fold result is retained as-is pending this decision.
