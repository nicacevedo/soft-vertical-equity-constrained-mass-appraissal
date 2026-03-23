#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=mit_normal # partition # ou_sloan_gpu #
#SBATCH --ntasks 1 # number of tasks
#SBATCH --cpus-per-task=64 # cpu
# SBATCH --gres=gpu:1  # THIS ONE WAS THE ONE MISSING
# SBATCH --mem-per-cpu=64GB # memory per cpu
#SBATCH --mem=164GB
# SBATCH --nodelist=node[1622, 3112]
#SBATCH --output=temp/logs/test_2026.out
#SBATCH --error=temp/logs/test_2026.err
#SBATCH -t 0-06:00:00 # time format is day-hours:minutes:seconds
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nacevedo@mit.edu
# SBATCH --array=1-22%10set
# mkdir -p temp/logs

# Activate your environment
# source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate fairness_env

# python run_temporal_cv.py
python quick_test_models.py \
  --rho-range 0.01,10 \
  --rho-count 10 \
  --rho-scale geom \
  --rho-group-range 1,1000 \
  --rho-group-count 5 \
  --rho-group-scale geom