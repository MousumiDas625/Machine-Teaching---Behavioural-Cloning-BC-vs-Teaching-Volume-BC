#!/bin/bash
#SBATCH --job-name=maze
#SBATCH --account=jessetho_1732
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/maze/data/log_%j.out

echo "===== maze STARTED: $(date) ====="
echo "Node: $SLURMD_NODENAME"

module load cuda

# Use full conda path — bypasses conda activate issues
PYTHON=/scratch1/mousumid/miniconda3/envs/machine_teaching/bin/python3

cd /project2/biyik_1165/mousumid/LIRA/machine

mkdir -p results/maze/data results/maze/plots results/maze/videos

$PYTHON -u run_game_pipeline.py --game maze --results_dir results/maze

echo "===== maze COMPLETE: $(date) ====="
