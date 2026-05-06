#!/bin/bash
#SBATCH --job-name=svm_sim2023
#SBATCH --partition=mit_normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96GB
#SBATCH --output=temp/logs/svm_sim2023.out
#SBATCH --error=temp/logs/svm_sim2023.err
#SBATCH -t 0-12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nacevedo@mit.edu

PY=/home/nacevedo/.conda/envs/fairness_env/bin/python

cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal

echo "[sim2023] starting run_temporal_cv.py at $(date)"
echo "[sim2023] hostname=$(hostname) cpus=$(nproc) python=$PY"

$PY run_temporal_cv.py \
    --data-path data/CCAO/2025/training_data_sim2023.parquet \
    --result-root output/rolling_origin_cv_sim2023 \
    --rho-values 0.1,30 --rho-count 30 --rho-scale geom \
    --sample-frac 0.95 \
    --n-bootstrap 0 \
    --parallel --parallel-cpu-fraction 0.90 --parallel-max-workers 28

ec=$?
echo "[sim2023] finished at $(date) with exit=$ec"
exit $ec
