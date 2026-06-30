import gymnasium as gym
import highway_env
import numpy as np
import random



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# -----------------------
# Same policy (kept identical for fair comparison)
# -----------------------
class RandomPolicy:
    def act(self, obs):
        return np.random.randint(5)


# -----------------------
# Propagator (VERY SIMPLE SAFETY FILTER)
# -----------------------
def propagator_check(obs, action):
    obs = np.array(obs).flatten()

    # ego features (highway-env standard)
    ego_speed = obs[3] if len(obs) > 3 else 0
    ego_x = obs[0] if len(obs) > 0 else 0
    ego_lane = obs[2] if len(obs) > 2 else 0

        # base risk increases with speed
    risk = ego_speed

        # lane change actions
    if action in [0, 2]:  # left/right
        # unsafe if going too fast
        if ego_speed > 0.25:
            return False, "lane change too fast"

        # add mild stochasticity ONLY if borderline
        if ego_speed > 0.15 and np.random.rand() < 0.2:
            return False, "borderline lane change risk"
        
        # acceleration safety
    if action == 3:  # accelerate
        if ego_speed > 0.35:
            return False, "too fast already"
        
        # deceleration always safe
    if action == 4:
        return True, "OK"
    
    return True, "OK"


def run(seed):

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

    for _ in range(10):

        action = policy.act(obs)
        action_counts[action] += 1

        safe, reason = propagator_check(obs, action)

        if not safe:
            rejected += 1
            action = 1  # fallback = keep lane

        obs, reward, terminated, truncated, info = env.step(action)
        speed = obs[0][3] if hasattr(obs, "__len__") else 0
        speed_sum += speed

        steps += 1

        if isinstance(info, dict) and info.get("crashed", False):
            crashes += 1

        if terminated or truncated:
            obs, info = env.reset(seed=seed)

        obs, info = env.reset()

        print(type(obs))
        print(np.array(obs).shape)
        print(obs)
            

    env.close()

    print("\n===== PROPAGATOR SYSTEM =====")
    print("steps:", steps)
    print("crashes:", crashes)
    print("crash rate:", crashes / max(1, steps))

    #propagator metrics
    print("rejected actions:", rejected)
    print("rejection rate:", rejected / max(1, steps))

    return (
        crashes / max(1, steps),
        crashes,
        action_counts / action_counts.sum(),
        speed_sum / max(1, steps),
        rejected / max(1, steps),
        rejected
    )


if __name__ == "__main__":

    seeds = list(range(2))

    crash_rates = []
    crashes_list = []
    action_distributions = []
    speeds = []
    rejected_rates = []
    rejected_list = []

    print("running propagator baseline ...")

    for s in seeds:
        crash_rate, crashes, action_dist, speed, rejected_rate, rejected = run(s)

        crash_rates.append(crash_rate)
        crashes_list.append(crashes)
        action_distributions.append(action_dist)
        speeds.append(speed)
        rejected_rates.append(rejected_rate)
        rejected_list.append(rejected)


    print("\n===== AVERAGED PROPAGATOR RESULTS =====")

    print("Crashes:")
    print("average crash rate:", np.mean(crash_rates))
    print("mean crashes:", np.mean(crashes_list))

    print("\nBehavior:")
    print("average speed:", np.mean(speeds))
    print("average action distribution:", np.mean(action_distributions, axis=0))

    print("\nRejections:")
    print("average rejection rate:", np.mean(rejected_rates))
    print("mean rejected actions:", np.mean(rejected_list))

