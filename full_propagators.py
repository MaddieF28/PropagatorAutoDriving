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


# =========================================================
# PROPAGATOR NETWORK
# Each propagator is a small independent function that reads
# only a few "cells" and writes a derived cell. If its inputs
# are missing (None), it leaves its output as None (unknown)
# instead of crashing or defaulting to unsafe/safe.
# =========================================================

def fresh_facts():
    return {
        "ego_speed": None,
        "front_gap": None,
        "front_rel_speed": None,
        "left_gap": None,
        "right_gap": None,
        "speed_limit": 30.0,
        "ttc": None,
        "speed_ok": None,
        "left_feasible": None,
        "right_feasible": None,
        "front_risk": None,
    }


def update_facts_from_obs(obs, facts, mask_prob=0.0):
    obs = np.array(obs).flatten().reshape(5, 5)  # 5 vehicles x [presence,x,y,vx,vy]
    ego = obs[0]
    others = obs[1:]

    ego_speed = float(ego[3])

    front_gap, front_rel_speed = None, None
    left_gap, right_gap = None, None

    LANE_EPS = 0.1  # same-lane threshold on normalized y

    for v in others:
        presence, x, y, vx, vy = v
        if presence < 0.5:
            continue
        if abs(y) < LANE_EPS and x > 0:
            if front_gap is None or x < front_gap:
                front_gap = float(x)
                front_rel_speed = float(ego[3] - vx)
        elif y < -LANE_EPS:
            if left_gap is None or abs(x) < left_gap:
                left_gap = float(abs(x))
        elif y > LANE_EPS:
            if right_gap is None or abs(x) < right_gap:
                right_gap = float(abs(x))

    candidates = {
        "ego_speed": ego_speed,
        "front_gap": front_gap,
        "front_rel_speed": front_rel_speed,
        "left_gap": left_gap,
        "right_gap": right_gap,
    }

    for key, value in candidates.items():
        if value is not None and np.random.rand() >= mask_prob:
            facts[key] = value
        else:
            facts[key] = None


# ---- individual propagators (each touches only its own cells) ----

def prop_ttc(facts):
    gap = facts["front_gap"]
    rel_speed = facts["front_rel_speed"]
    if gap is None or rel_speed is None:
        facts["ttc"] = None
        return
    if rel_speed <= 0:
        facts["ttc"] = float("inf")
    else:
        facts["ttc"] = gap / rel_speed


def prop_front_risk(facts):
    ttc = facts["ttc"]
    if ttc is None:
        facts["front_risk"] = None
        return
    facts["front_risk"] = ttc < 2.0


def prop_speed_ok(facts):
    speed = facts["ego_speed"]
    if speed is None:
        facts["speed_ok"] = None
        return
    facts["speed_ok"] = speed <= facts["speed_limit"]


def prop_left_feasible(facts):
    gap = facts["left_gap"]
    if gap is None:
        facts["left_feasible"] = None
        return
    facts["left_feasible"] = gap >= 8.0


def prop_right_feasible(facts):
    gap = facts["right_gap"]
    if gap is None:
        facts["right_feasible"] = None
        return
    facts["right_feasible"] = gap >= 8.0


PROPAGATORS = [prop_ttc, prop_front_risk, prop_speed_ok, prop_left_feasible, prop_right_feasible]


def run_propagators(facts):
    # order-independent in principle (each only depends on raw cells),
    # so a single pass is enough here, but loop kept explicit for clarity
    for p in PROPAGATORS:
        p(facts)


def feasible_actions(facts):
    """Given current facts, return the set of actions NOT contradicted by
    a known constraint. Unknown facts default to 'not blocked' (treated as
    permissive-unknown) rather than silently unsafe or silently safe-blocked."""
    feasible = {0, 1, 2, 3, 4}  # left, keep, right, accel, brake

    if facts["left_feasible"] is False:
        feasible.discard(0)
    if facts["right_feasible"] is False:
        feasible.discard(2)
    if facts["front_risk"] is True:
        feasible.discard(3)  # don't accelerate into risk
    if facts["speed_ok"] is False:
        feasible.discard(3)  # don't accelerate past limit

    return feasible


def known_fraction(facts):
    keys = ["ego_speed", "front_gap", "front_rel_speed", "left_gap", "right_gap"]
    known = sum(1 for k in keys if facts[k] is not None)
    return known / len(keys)


# -----------------------
# Main experiment loop
# -----------------------
def run(seed, mask_prob=0.0):
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
    known_frac_sum = 0

    for _ in range(300):

        action = policy.act(obs)
        action_counts[action] += 1

        facts = fresh_facts()
        update_facts_from_obs(obs, facts, mask_prob=mask_prob)
        run_propagators(facts)

        allowed = feasible_actions(facts)
        known_frac_sum += known_fraction(facts)

        if action not in allowed:
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
        
       

    env.close()

    print(f"\nRun {seed + 1}")
    print("steps:", steps)
    print("crashes:", crashes)
    print("crash rate:", crashes / max(1, steps))
    print("action distribution:", action_counts / action_counts.sum())
    print("avg speed:", speed_sum / max(1, steps))
    print("rejected actions:", rejected)
    print("rejection rate:", rejected / max(1, steps))
    print("avg known-fact fraction:", known_frac_sum / max(1, steps))

    return (
        crashes / max(1, steps),
        crashes,
        action_counts / action_counts.sum(),
        speed_sum / max(1, steps),
        rejected / max(1, steps),
        rejected,
        known_frac_sum / max(1, steps),
    )


if __name__ == "__main__":

    seeds = list(range(15))

    # set MASK_PROB > 0 to simulate sensor dropout and test graceful degradation
    MASK_PROB = 0.3

    crash_rates = []
    crashes_list = []
    action_distributions = []
    speeds = []
    rejected_rates = []
    rejected_list = []
    known_fracs = []

    print("running propagator network ...")
    print(f"mask probability: {MASK_PROB}")

    for s in seeds:
        crash_rate, crashes, action_dist, speed, rejected_rate, rejected, known_frac = run(
            s, mask_prob=MASK_PROB
        )

        crash_rates.append(crash_rate)
        crashes_list.append(crashes)
        action_distributions.append(action_dist)
        speeds.append(speed)
        rejected_rates.append(rejected_rate)
        rejected_list.append(rejected)
        known_fracs.append(known_frac)

    print("\n===== AVERAGED PROPAGATOR NETWORK RESULTS =====")

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

    print("\nPartial information:")
    print("average known-fact fraction:", np.mean(known_fracs))

    