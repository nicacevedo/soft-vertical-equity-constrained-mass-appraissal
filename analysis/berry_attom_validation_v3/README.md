# Berry/ATTOM validation v3

> **Read `QUARANTINE_REVIEW.md` first.** `reports/FINAL_V3_REPORT.md` remains the
> execution/provenance report; `QUARANTINE_REVIEW.md` governs the canonical interpretation of those
> results — which artifacts are protocol-conforming, which are qualified, which are exploratory, and
> which must not be used as scientific claims.

Isolated follow-on to frozen `analysis/berry_cmf_validation/` (v1) and
`analysis/berry_attom_validation_v2/` (v2). **Do not overwrite either.**

v2 is superseded because freeze logic could use held-out test performance.
v3 forbids scoring the chronological test block until
`panel_freeze/final_panel_freeze_v3.yaml` exists.

This is **not** that unrelated copied-prompt task. It is Berry/ATTOM mass-appraisal validation.

Protocol: `protocol_v3.yaml` (frozen before linkage/baseline outcomes).

```
/home/nacevedo/.conda/envs/fairness_env/bin/python analysis/berry_attom_validation_v3/scripts/copy_v2_provenance.py
# then Sloan DAG:
bash analysis/berry_attom_validation_v3/slurm/submit_pipeline.sh
```

Partition: `sched_mit_sloan_batch_r8` (Sloan). `sched_mit_sloan_batch` compute nodes cannot load fairness_env LightGBM (GLIBC_2.27). Never default to `mit_normal`.
Direct/Surrogate are **not** submitted until freeze authorizes them.
