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
        left_clear = True
        right_clear = True

        for v in others:
            presence, x, y, vx, vy = v
            if presence < 0.5:
                continue
            if abs(y) < 0.12 and x > 0:
                if front_gap is None or x < front_gap:
                    front_gap = x
            elif -0.37 < y < -0.12 and abs(x) < 0.25:
                left_clear = False
            elif 0.12 < y < 0.37 and abs(x) < 0.25:
                right_clear = False

        front_desc = f"{front_gap:.2f} (CLOSE)" if front_gap and front_gap < 0.3 else (f"{front_gap:.2f}" if front_gap else "none")

        return f"""You are a highway driving agent. Pick the safest and most efficient action.

Current state:
- Your speed: {ego_speed:.2f} (max safe speed is 0.40)
- Car ahead gap: {front_desc}
- Left lane clear: {"yes" if left_clear else "no"}
- Right lane clear: {"yes" if right_clear else "no"}

Rules:
- If speed is above 0.30 do NOT accelerate
- If car ahead gap is CLOSE change lanes or brake
- If left or right lane is clear and car is ahead, change lanes

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

    env = gym.make("highway-v0", render_mode= None)
    env.action_space.seed(seed)
    obs, info = env.reset(seed = seed)
    

    policy = LLMPolicy()

    steps = 0
    crashes = 0

    action_counts = np.zeros(5)

    for _ in range(50):

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