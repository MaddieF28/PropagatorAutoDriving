import gymnasium as gym
import highway_env
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# -----------------------
# Simple NN policy (untrained = random weights but structured)
# -----------------------
class RandomPolicy:
    def act(self, obs):
        return np.random.randint(5)  # 0–4 actions


def run(seed):
    speed_sum = 0

    set_seed(seed)

    env = gym.make("highway-v0", render_mode=None)
    env.action_space.seed(seed)
    obs, info = env.reset(seed = seed)
    

    policy = RandomPolicy()

    steps = 0
    crashes = 0

    action_counts = np.zeros(5)

    for _ in range(60):

        action = policy.act(obs)
        action_counts[action] += 1

        obs, reward, terminated, truncated, info = env.step(action)
        speed = obs[0][3] if hasattr(obs, "__len__") else 0

        steps += 1

        if isinstance(info, dict) and info.get("crashed", False):
            crashes += 1

        if terminated or truncated:
            obs, info = env.reset(seed=seed)

        speed_sum += speed

    env.close()

    print(f"\nRun {seed+1}")
    print("steps:", steps)
    print("crashes:", crashes)
    print("crash rate:", crashes / max(1, steps))
    print("action distribution:", action_counts / action_counts.sum())
    print("avg speed:", speed_sum / max(1, steps))

    return (
        crashes / max(1, steps),
        crashes,
        action_counts / action_counts.sum(),
        speed_sum / max(1, steps)
    )

  


if __name__ == "__main__":

    seeds = list(range(5))
    crash_rates = []
    crash_counts = []
    action_distributions = []
    speeds = []


    print("\nrunning ...")

    print("\n===== BASELINE =====")

    for s in seeds:
        crash_rate, crashes, action_dist, speed = run(s)
        crash_rates.append(crash_rate)
        crash_counts.append(crashes)
        action_distributions.append(action_dist)
        speeds.append(speed)

    print("\n===== AVERAGED RESULTS =====")

    print("mean crash count:", np.mean(crash_counts))
    print("std crash count:", np.std(crash_counts))


    print("average crash rate:", np.mean(crash_rates))
    print("average speed:", np.mean(speeds))
    print("average action distribution:", np.mean(action_distributions, axis=0))