#!/bin/bash
#SBATCH --job-name=tr994_stage
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v6_preselection_994/analysis/data_id=d4929d43ec19badf/split_id=3d464d4a611b131b/penalty_path_analysis/transition_regions_v1/logs/%x_%j.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v6_preselection_994/analysis/data_id=d4929d43ec19badf/split_id=3d464d4a611b131b/penalty_path_analysis/transition_regions_v1/logs/%x_%j.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "$NUMBA_CACHE_DIR"

STAGE="${TR_STAGE:?TR_STAGE must be set}"
RESULT_ROOT="${TR_RESULT_ROOT:-output/paper_v6_preselection_994}"
OUTPUT_ROOT="${TR_OUTPUT_ROOT:-output/paper_v6_preselection_994/analysis/data_id=d4929d43ec19badf/split_id=3d464d4a611b131b/penalty_path_analysis/transition_regions_v1}"
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/qa" "$OUTPUT_ROOT/cluster"

echo "TR994 stage=$STAGE job=${SLURM_JOB_ID:-} partition=${SLURM_JOB_PARTITION:-} node=${SLURMD_NODENAME:-}"
echo "result_root=$RESULT_ROOT"
echo "output_root=$OUTPUT_ROOT"
PY="/home/nacevedo/.conda/envs/fairness_env/bin/python"
echo "python=$PY"
"$PY" -V

"$PY" scripts/analyze_transition_regions.py \
  --stage "$STAGE" \
  --result-root "$RESULT_ROOT" \
  --output-root "$OUTPUT_ROOT"

echo "TR994_DONE stage=$STAGE job=${SLURM_JOB_ID:-}"
