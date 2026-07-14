import gymnasium as gym
import highway_env
import numpy as np

import random
import requests
import re


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


##---------
## ollama policy
## phi4-mini
##--------
class LLMPolicy:
    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.model = "phi4-mini"

    def format_obs(self, obs):
        obs = np.array(obs).reshape(5, 5)
        ego = obs[0]
        others = obs[1:]

        ego_speed = ego[3]

        front_gap = None
        front_rel_speed = None
        left_gap = None
        right_gap = None

        for v in others:
            presence, x, y, vx, vy = v
            if presence < 0.5:
                continue
            if abs(y) < 0.12 and x > 0:
                if front_gap is None or x < front_gap:
                    front_gap = x
                    front_rel_speed = ego_speed - vx
            elif -0.37 < y < -0.12:
                if left_gap is None or abs(x) < left_gap:
                    left_gap = abs(x)
            elif 0.12 < y < 0.37:
                if right_gap is None or abs(x) < right_gap:
                    right_gap = abs(x)

        # compute TTC
        if front_gap is not None and front_rel_speed is not None and front_rel_speed > 0:
            ttc = front_gap / front_rel_speed
        else:
            ttc = None

        # label helpers
        def gap_label(g):
            if g is None:
                return "none (clear)"
            if g < 0.15:
                return f"{g:.2f} (UNSAFE - too close)"
            if g < 0.25:
                return f"{g:.2f} (tight - risky)"
            if g < 0.4:
                return f"{g:.2f} (possible)"
            return f"{g:.2f} (clear)"

        def ttc_label(t):
            if t is None:
                return "no car ahead"
            if t < 1.5:
                return f"{t:.1f}s (CRITICAL)"
            if t < 3.0:
                return f"{t:.1f}s (warning)"
            return f"{t:.1f}s (safe)"

        def speed_label(s):
            if s > 0.35:
                return f"{s:.2f} (too fast)"
            if s < 0.15:
                return f"{s:.2f} (too slow)"
            return f"{s:.2f} (normal)"

        return f"""You are a highway driving agent. Choose the safest and most efficient action.

        Current state:
        - Speed: {speed_label(ego_speed)}
        - Front gap: {gap_label(front_gap)}
        - Front TTC: {ttc_label(ttc)}
        - Left lane gap: {gap_label(left_gap)}
        - Right lane gap: {gap_label(right_gap)}

        Decision rules:
        - If TTC is CRITICAL or front gap is UNSAFE → brake or change lanes immediately
        - If left or right gap is possible or clear and front is dangerous → change lanes
        - If speed is too fast → brake
        - If speed is too slow and front is safe → accelerate
        - Otherwise → keep lane

        Actions:
        0 = change left
        1 = keep lane
        2 = change right
        3 = accelerate
        4 = brake

        Reply with a single digit 0-4 and absolutely nothing else."""

    def act(self, obs):
        prompt = self.format_obs(obs)
        try:
            response = requests.post(self.url, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.5}
            })
            text = response.json()["response"].strip()
            match = re.search(r"[0-4]", text)
            if match:
                return int(match.group())
            return 1  # fallback keep lane
        except Exception:
            return 1  # fallback on any error


def run(seed):
    speed_sum = 0

    set_seed(seed)


    env = gym.make("highway-v0", render_mode=None)
    env.unwrapped.configure({
        "vehicles_count": 100,
        "vehicles_density": 3,
        "ego_spacing": 1,
        "lanes_count": 3,
        "duration": 40,
    })
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    

    policy = LLMPolicy()

    steps = 0
    crashes = 0

    action_counts = np.zeros(5)

    for _ in range(100):

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

    seeds = list(range(4))
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