import gymnasium as gym
import highway_env
import numpy as np
import random
import requests
import re


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

class Waypoint:
    def __init__(self, x, lane):
        self.x = x          # meters ahead
        self.lane = lane    # target lane


##---------
## ollama policy
## phi4-mini
##--------
class LLMPolicy:
    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.model = "phi4-mini"

    def format_obs(self, obs, goal, distance, lane_error, ego_lane):
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

            if ego_lane > goal.lane:
                lane_guidance = "You are right of the goal. Prioritize action 0 (change left) to reach your destination."
            elif ego_lane < goal.lane:
                lane_guidance = "You are left of the goal. Prioritize action 2 (change right) to reach your destination."


            return f"""You are a highway driving agent. Your objective is to safely reach the waypoint.

    Priorities:
    1. Never collide.
    2. Reach the waypoint.
    3. Maintain speed within .1 of the speed limit by using brake or acceleration
    4. Avoid a front risk by using brake or change lane

    Current state:
    - Your speed: {ego_speed:.2f} (speed limit = .30)
    - Car ahead gap: {front_desc}
    - Left lane clear: {"yes" if left_clear else "no"}
    - Right lane clear: {"yes" if right_clear else "no"}
    - Your current lane: {ego_lane}

    Waypoint:
    You have {distance:.1f} meters to reach lane {goal.lane}.
    {lane_guidance}

    Actions:
    0 = change left
    1 = keep lane
    2 = change right
    3 = accelerate
    4 = brake

    Reply with a single digit 0-4 and absolutely nothing else."""
    
    def act(self, obs, goal, distance, lane_error, ego_lane):
        prompt = self.format_obs(obs, goal, distance, lane_error, ego_lane)
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



def fresh_facts():
    return {
        # --- safety/physical facts (existing) ---
        "ego_speed": None,
        "front_gap": None,
        "front_rel_speed": None,
        "left_gap": None,
        "right_gap": None,
        "speed_limit": .35,
        "ttc": None,
        "speed_slow": None,
        "speed_fast": None,
        "left_feasible": None,
        "right_feasible": None,
        "front_risk": None,

        # --- mission facts (new) ---
        "goal_distance": None,
        "goal_lane": None,
        "lane_error": None,
        "needs_left": None,
        "needs_right": None,
        "needs_speed_up": None,
        "needs_slow_down": None,
        "in_goal_lane": None,
    }


def update_facts_from_obs(obs, facts, mask_prob=0.0):
    obs = np.array(obs).flatten().reshape(5, 5)  # 5 ve hicles x [presence,x,y,vx,vy]
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
    facts["front_risk"] = ttc < .5


def prop_speed_slow(facts):
    speed = facts["ego_speed"]
    if speed is None:
        facts["speed_slow"] = None
        return
    facts["speed_slow"] = (speed <= (facts["speed_limit"] - .1))

def prop_speed_fast(facts):
    speed = facts["ego_speed"]
    if speed is None:
        facts["speed_fast"] = None
        return
    facts["speed_fast"] = speed >= facts["speed_limit"]


def prop_left_feasible(facts):
    gap = facts["left_gap"]
    if gap is None:
        facts["left_feasible"] = None
        return
    facts["left_feasible"] = gap >= .15


def prop_right_feasible(facts):
    gap = facts["right_gap"]
    if gap is None:
        facts["right_feasible"] = None
        return
    facts["right_feasible"] = gap >= .15


PROPAGATORS = [prop_ttc, prop_front_risk, prop_speed_slow, prop_speed_fast, prop_left_feasible, prop_right_feasible]


def run_propagators(facts):
    for p in PROPAGATORS:
        p(facts)


def feasible_actions(facts):
    feasible = {0, 1, 2, 3, 4}  # left, keep, right, accel, brake

    if facts["left_feasible"] is False:
        feasible.discard(0)
    if facts["right_feasible"] is False:
        feasible.discard(2)
        feasible.discard(3)  
        feasible.discard(1)
    if facts["speed_slow"] is True:
        feasible.discard(4) 
        feasible.discard(1)
    if facts["speed_fast"] is True:
        feasible.discard(1)
        feasible.discard(3)

    return feasible


def known_fraction(facts):
    keys = ["ego_speed", "front_gap", "front_rel_speed", "left_gap", "right_gap"]
    known = sum(1 for k in keys if facts[k] is not None)
    return known / len(keys)



def run(seed, mask_prob=0.0):
    set_seed(seed)


    env = gym.make('highway-v0', 
        render_mode = "human",
        config={
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": [0, 5, 10, 15, 20, 25, 30, 35],
        },
        "vehicles_density" : .1,
        "lanes_count": 4,
    })

    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    ego_start_x = env.unwrapped.vehicle.position[0]

    waypoints = [
        Waypoint(ego_start_x + 100, 0), 
        Waypoint(ego_start_x + 200, 3), 
        Waypoint(ego_start_x + 300, 0),  
    ]

    current_waypoint = 0


    policy = LLMPolicy()

    steps = 0
    crashes = 0
    rejected = 0
    action_counts = np.zeros(5)
    speed_sum = 0
    known_frac_sum = 0



    for _ in range(30):

        ego = env.unwrapped.vehicle
        ego_x = ego.position[0]
        ego_lane = ego.lane_index[2]

        goal = waypoints[current_waypoint]
        distance = abs(goal.x - ego_x)

        if ego_x >= goal.x or distance < 4:
            if current_waypoint < len(waypoints) - 1:
                print(f"\nReached Waypoint: {True if ego_lane == waypoints[current_waypoint].lane else False}")
                current_waypoint += 1
                print(f"-- Waypoint updated to {current_waypoint} ---")

                # --- refresh goal/distance to the NEW waypoint ---
                goal = waypoints[current_waypoint]
                distance = abs(goal.x - ego_x)
            else:
                pass

        lane_error = goal.lane - ego_lane 

        llmaction = policy.act(obs, goal, distance, lane_error, ego_lane)  # sees correct goal
        action = llmaction
        action_counts[action] += 1

        print(f"ego location: {ego_x}")
        print(f"distance: {distance}")
        print(f"Current lane: {ego_lane}")
        print(f"Goal lane: {goal.lane}")


        facts = fresh_facts()
        update_facts_from_obs(obs, facts, mask_prob=mask_prob)
        run_propagators(facts)

        facts["goal_distance"] = distance
        facts["goal_lane"] = goal.lane
        facts["lane_error"] = lane_error

        allowed = feasible_actions(facts)
        known_frac_sum += known_fraction(facts)

        if action not in allowed:
            rejected += 1
            action = 4  # fallback = keep lane

 

        obs, reward, terminated, truncated, info = env.step(action)
        speed = obs[0][3] if hasattr(obs, "__len__") else 0
        speed_sum += speed

        steps += 1

        if isinstance(info, dict) and info.get("crashed", False):
            crashes += 1

        if terminated or truncated:
            obs, info = env.reset(seed=seed)
        
       
        print(f"\nLLM Proposed: {llmaction}, allowed: {allowed} | front_risk: {facts['front_risk']} | ttc: {facts['ttc']} | front_gap: {facts['front_gap']}")
        print(f"  final action: {action}")

        print(f"  raw vehicle speed: {env.unwrapped.vehicle.speed:.3f}")
        print(f"  crashed this step: {info.get('crashed')} | speed: {obs[0][3]:.3f}")


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

    seeds = list(range(5))

    # set MASK_PROB > 0 to simulate sensor dropout and test graceful degradation
    MASK_PROB = 0

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

    