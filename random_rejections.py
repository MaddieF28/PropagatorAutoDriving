import gymnasium as gym
import highway_env
import numpy as np
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# -----------------------
# Same policy as baseline/propagator scripts (kept identical for fair comparison)
# -----------------------
class RandomPolicy:
    def act(self, obs):
        return np.random.randint(5)  # 0=left, 1=keep, 2=right, 3=accelerate, 4=brake


# Set this to match the propagator network's observed rejection rate (~0.41)
REJECT_PROB = 0.41


def run(seed, reject_prob=REJECT_PROB):
    set_seed(seed)

    env = gym.make("highway-v0", render_mode=None)
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    policy = RandomPolicy()

    steps = 0
    crashes = 0
    rejected = 0
    action_counts = np.zeros(5)
    speed_sum = 0

    for _ in range(60):

        action = policy.act(obs)
        action_counts[action] += 1

        # flat random rejection, no state, no reasoning
        if np.random.rand() < reject_prob:
            rejected += 1
            action = 1  # same fallback as propagator version: keep lane

        obs, reward, terminated, truncated, info = env.step(action)
        speed = obs[0][3] if hasattr(obs, "__len__") else 0
        speed_sum += speed

        steps += 1

        if isinstance(info, dict) and info.get("crashed", False):
            crashes += 1

        if terminated or truncated:
            obs, info = env.reset(seed=seed)

    env.close()

    print(f"\nRun {seed + 1}")
    print("steps:", steps)
    print("crashes:", crashes)
    print("crash rate:", crashes / max(1, steps))
    print("action distribution:", action_counts / action_counts.sum())
    print("avg speed:", speed_sum / max(1, steps))
    print("rejected actions:", rejected)
    print("rejection rate:", rejected / max(1, steps))

    return (
        crashes / max(1, steps),
        crashes,
        action_counts / action_counts.sum(),
        speed_sum / max(1, steps),
        rejected / max(1, steps),
        rejected,
    )


if __name__ == "__main__":

    seeds = list(range(5))

    crash_rates = []
    crashes_list = []
    action_distributions = []
    speeds = []
    rejected_rates = []
    rejected_list = []

    print("running flat-random-rejection control ...")
    print(f"reject probability: {REJECT_PROB}")

    for s in seeds:
        crash_rate, crashes, action_dist, speed, rejected_rate, rejected = run(s)

        crash_rates.append(crash_rate)
        crashes_list.append(crashes)
        action_distributions.append(action_dist)
        speeds.append(speed)
        rejected_rates.append(rejected_rate)
        rejected_list.append(rejected)

    print("\n===== AVERAGED RANDOM-REJECTION CONTROL RESULTS =====")

    print("\nCrashes:")
    print("average crash rate:", np.mean(crash_rates))
    print("mean crashes:", np.mean(crashes_list))
    print("std crashes:", np.std(crashes_list))

    print("\nBehavior:")
    print("average speed:", np.mean(speeds))
    print("average action distribution:", np.mean(action_distributions, axis=0))

    print("\nRejections:")
    print("average rejection rate:", np.mean(rejected_rates))
    print("mean rejected actions:", np.mean(rejected_list))