#!/bin/bash
#SBATCH --job-name=maze_train
#SBATCH --account=jessetho_1732
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=training_log_%j.out

# Load necessary modules
module load cuda

# Activate your environment
eval "$(conda shell.bash hook)"
conda activate machine_teaching

# Run your script
python3 experiments/train_expert.py

# --- STEP 2: Collect Demonstrations ---
echo "======================================"
echo "Starting Step 2: Collecting Demonstrations"
echo "======================================"
python3 experiments/collect_maze_demos.py

# --- STEP 3: Run TV-BC vs BC Experiment ---
echo "======================================"
echo "Starting Step 3: Running Generalisation Experiment"
echo "======================================"
python3 experiments/run_maze_generalisation.py

# --- STEP 6: Record Videos ---
echo "======================================"
echo "Starting Step 6: Recording Videos"
echo "======================================"
python3 experiments/record_videos.py

echo "PIPELINE COMPLETE!"
