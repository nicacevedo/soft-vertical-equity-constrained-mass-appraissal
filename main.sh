#!/bin/bash
#SBATCH --job-name=CV_new
#SBATCH --partition=mit_normal # partition # ou_sloan_gpu #
#SBATCH --ntasks 1 # number of tasks
#SBATCH --cpus-per-task=64 # cpu
# SBATCH --gres=gpu:1  # THIS ONE WAS THE ONE MISSING
# SBATCH --mem-per-cpu=64GB # memory per cpu
#SBATCH --mem=164GB
# SBATCH --nodelist=node[1622, 3112]
#SBATCH --output=temp/logs/CV_new_2026.out
#SBATCH --error=temp/logs/CV_new_2026.err
#SBATCH -t 0-12:00:00 # time format is day-hours:minutes:seconds
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nacevedo@mit.edu
# SBATCH --array=1-22%10set
# mkdir -p temp/logs

# Activate your environment
# source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate fairness_env

python run_temporal_cv.py