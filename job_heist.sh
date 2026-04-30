#!/bin/bash
#SBATCH --job-name=heist
#SBATCH --account=jessetho_1732
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/heist/data/log_%j.out

echo "===== heist STARTED: $(date) ====="
echo "Node: $SLURMD_NODENAME"

module load cuda

# Use full conda path — bypasses conda activate issues
PYTHON=/scratch1/mousumid/miniconda3/envs/machine_teaching/bin/python3

cd /project2/biyik_1165/mousumid/LIRA/machine

mkdir -p results/heist/data results/heist/plots results/heist/videos

$PYTHON -u run_game_pipeline.py --game heist --results_dir results/heist

echo "===== heist COMPLETE: $(date) ====="
