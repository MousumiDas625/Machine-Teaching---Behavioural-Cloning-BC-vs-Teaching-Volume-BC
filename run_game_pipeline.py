import argparse
import os
import sys
import subprocess

def patch_and_run(script_path, replacements, game, results_dir):
    with open(script_path) as f:
        code = f.read()
    for old, new in replacements:
        code = code.replace(old, new)
    tmp = f"/tmp/patched_{game}_{os.path.basename(script_path)}"
    with open(tmp, 'w') as f:
        f.write("import sys; sys.path.insert(0, '/project2/biyik_1165/mousumid/LIRA/machine')\n" + code)
    subprocess.run(['/scratch1/mousumid/miniconda3/envs/machine_teaching/bin/python3', '-u', tmp], check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game',        default='maze')
    parser.add_argument('--results_dir', default='results/maze')
    args = parser.parse_args()

    game        = args.game
    results_dir = args.results_dir

    os.makedirs(f"{results_dir}/data",   exist_ok=True)
    os.makedirs(f"{results_dir}/plots",  exist_ok=True)
    os.makedirs(f"{results_dir}/videos", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  GAME: {game.upper()}")
    print(f"  Results dir: {results_dir}")
    print(f"{'='*60}\n")

    all_replacements = [
        ("env_name          = 'maze'",    f"env_name          = '{game}'"),
        ("env_name='maze'",               f"env_name='{game}'"),
        ("env_name = 'maze'",             f"env_name = '{game}'"),
        ("results/data/maze_expert_best", f"{results_dir}/data/{game}_expert_best"),
        ("results/data/maze_expert",      f"{results_dir}/data/{game}_expert"),
        ("results/data/expert_training",  f"{results_dir}/data/expert_training"),
        ("results/data/maze_demos",       f"{results_dir}/data/{game}_demos"),
        ("results/data/maze_generalisation", f"{results_dir}/data/{game}_generalisation"),
        ("results/data/bc_policy_maze",   f"{results_dir}/data/bc_policy_{game}"),
        ("results/data/tvbc_policy_maze", f"{results_dir}/data/tvbc_policy_{game}"),
        ("results/plots/maze_generalisation", f"{results_dir}/plots/{game}_generalisation"),
        ("results/videos/",               f"{results_dir}/videos/"),
        ("EXPERT_PATH = \"results/data/maze_expert_best.pt\"",
         f"EXPERT_PATH = \"{results_dir}/data/{game}_expert_best.pt\""),
        ("EXPERT_PATH = 'results/data/maze_expert_best.pt'",
         f"EXPERT_PATH = '{results_dir}/data/{game}_expert_best.pt'"),
    ]

    print(f"Step 1: Training expert for {game}...")
    patch_and_run('experiments/train_expert.py',
                  all_replacements, game, results_dir)

    print(f"Step 2: Collecting demos for {game}...")
    patch_and_run('experiments/collect_maze_demos.py',
                  all_replacements, game, results_dir)

    print(f"Step 3: Running TV-BC vs BC for {game}...")
    patch_and_run('experiments/run_maze_generalisation.py',
                  all_replacements, game, results_dir)

    print(f"Step 4: Recording videos for {game}...")
    os.system("pip install imageio imageio-ffmpeg -q")
    patch_and_run('experiments/record_videos.py',
                  all_replacements, game, results_dir)

    print(f"\n{'='*60}")
    print(f"  COMPLETE: {game.upper()}")
    print(f"  Results: {results_dir}/")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
