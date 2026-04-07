# test_tvbc.py
import sys
sys.path.insert(0, ".")

import numpy as np
import pickle
from agents.policy_net       import PolicyNetwork
from agents.rl_teacher       import RLTeacher
from gridworld_env.gridworld import GridWorld
from learners.tv_bc          import TeacherAwareBC


# Load data
with open("data/rollouts/teacher.pkl",       "rb") as f:
    teacher = pickle.load(f)
with open("data/rollouts/omega_star.pkl",    "rb") as f:
    omega_star = pickle.load(f)
with open("data/selected/train_states.pkl",  "rb") as f:
    states = pickle.load(f)
with open("data/selected/train_actions.pkl", "rb") as f:
    actions = pickle.load(f)

env = GridWorld(grid_size=8, omega_star=omega_star,
                max_steps=50, seed=42)

def evaluate_policy(policy, env, n_episodes=20, seed=99):
    import numpy as np
    rng          = np.random.default_rng(seed)
    total_reward = 0.0
    policy.eval()
    for _ in range(n_episodes):
        state = env.reset(start_pos=int(rng.integers(0, env.n_states)))
        done  = False
        ep_reward = 0.0
        while not done:
            probs  = policy.get_action_probs(state).numpy()
            action = int(np.argmax(probs))
            state, reward, done, _ = env.step(action)
            ep_reward += reward
        total_reward += ep_reward
    return total_reward / n_episodes

# Build fresh policy with SAME seed as BC test — fair comparison
policy  = PolicyNetwork(state_dim=64, action_dim=4, hidden_dim=64, seed=0)
eval_fn = lambda p: evaluate_policy(p, env, n_episodes=20)

print(f"Initial return: {evaluate_policy(policy, env):.3f}")
print(f"Teacher return: ~47.9\n")

tvbc = TeacherAwareBC(
    policy  = policy,
    teacher = teacher,
    eta     = 0.001,
    beta    = 2.0,
    seed    = 7
)

tvbc.train(
    states     = states,
    actions    = actions,
    n_epochs   = 50,
    batch_size = 20,
    eval_fn    = eval_fn,
    eval_every = 10,
    verbose    = True
)

final_return = evaluate_policy(policy, env, n_episodes=50)
print(f"\nFinal TV-BC return: {final_return:.3f}")
print(f"BC return was:      45.268  (from previous test)")
print(f"Difference: {final_return - 45.268:+.3f}")

# Action probs at best cell
import numpy as np
state_5 = np.zeros(64, dtype=np.float32)
state_5[5] = 1.0
probs = policy.get_action_probs(state_5).numpy()
print(f"\nTV-BC action probs at cell 5:")
for name, p in zip(["UP","DOWN","LEFT","RIGHT"], probs):
    print(f"  {name:5s}: {p:.4f}")
