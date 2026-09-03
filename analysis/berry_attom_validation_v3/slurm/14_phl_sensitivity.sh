#!/bin/bash
# Philadelphia broader-residential-use SENSITIVITY cohort.
#
# The frozen primary cohort keeps PROPERTYUSESTANDARDIZED=385 only, which drops
# 82% of Philadelphia's safe-history sales with a hump-shaped price-decile
# profile. This job probes that filter. It is NOT a freeze revision: it writes
# to its own table and its own baselines directory, scores VALIDATION ONLY, and
# never runs Direct or Surrogate.
#SBATCH --job-name=v3_phlsens
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --output=analysis/berry_attom_validation_v3/logs/phlsens_%j.out
#SBATCH --error=analysis/berry_attom_validation_v3/logs/phlsens_%j.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
CPUS="${SLURM_CPUS_PER_TASK:-16}"
# dcor/numba JIT cache on node-local disk. Job 21882241_0 (Wayne surrogate)
# died with OSError errno 116 "Stale file handle" reading the numba index off
# the shared filesystem, which cancelled the downstream afterok chain.
NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID:-0}"
mkdir -p "${NUMBA_CACHE_DIR}"
export NUMBA_CACHE_DIR
trap 'rm -rf "${NUMBA_CACHE_DIR}"' EXIT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${CPUS}"
echo "phl_sensitivity cpus=${CPUS} host=$(hostname) partition=${SLURM_JOB_PARTITION:-} start=$(date -Is)"

"${PY}" analysis/berry_attom_validation_v3/scripts/build_modeling_tables.py \
  --county-key philadelphia --property-use-set broad_residential

"${PY}" analysis/berry_attom_validation_v3/scripts/run_prefreeze_baselines.py \
  --county-key philadelphia --property-use-set broad_residential --lgbm-threads "${CPUS}"

# Guard the label rather than trusting it: the sensitivity cohort must not have
# produced any held-out or positive-rho artifact.
SENS="analysis/berry_attom_validation_v3/baselines_pre_freeze/philadelphia_broad_residential_sensitivity"
if compgen -G "${SENS}/heldout*" > /dev/null || compgen -G "${SENS}/direct*" > /dev/null \
   || compgen -G "${SENS}/surrogate*" > /dev/null; then
  echo "FATAL: sensitivity cohort produced held-out or positive-rho artifacts" >&2
  exit 1
fi
echo "done $(date -Is)"
