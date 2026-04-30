#!/bin/bash
#SBATCH --job-name=coinrun
#SBATCH --account=jessetho_1732
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/coinrun/data/log_%j.out

echo "===== coinrun STARTED: $(date) ====="
echo "Node: $SLURMD_NODENAME"

module load cuda

# Use full conda path — bypasses conda activate issues
PYTHON=/scratch1/mousumid/miniconda3/envs/machine_teaching/bin/python3

cd /project2/biyik_1165/mousumid/LIRA/machine

mkdir -p results/coinrun/data results/coinrun/plots results/coinrun/videos

$PYTHON -u run_game_pipeline.py --game coinrun --results_dir results/coinrun

echo "===== coinrun COMPLETE: $(date) ====="
