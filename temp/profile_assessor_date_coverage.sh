#!/bin/bash
#SBATCH --job-name=assessor_dates
#SBATCH --partition=sched_mit_sloan_interactive
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=temp/logs/assessor_dates_%j.out
#SBATCH --error=temp/logs/assessor_dates_%j.err

set -eo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
mkdir -p temp/logs
/home/nacevedo/.conda/envs/fairness_env/bin/python -u temp/profile_assessor_date_coverage.py
