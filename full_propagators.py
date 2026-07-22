import gymnasium as gym
import highway_env
import numpy as np
import random
import requests
import re
import csv
import os




SPEED_LIMIT = 30  # real speed units, matches TARGET_SPEEDS

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

    def format_obs(self, obs, goal, distance, lane_error, ego_lane, speed_limit, ego_speed, num_lanes):
        obs = np.array(obs).reshape(5, 5)
        ego = obs[0]
        others = obs[1:]

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

        at_left_edge = ego_lane <= 0
        at_right_edge = ego_lane >= num_lanes - 1

        boundary_note = ""
        if at_left_edge:
            boundary_note = "You are already in the leftmost lane — action 0 (change left) is not possible."
        elif at_right_edge:
            boundary_note = "You are already in the rightmost lane — action 2 (change right) is not possible."

        if ego_lane > goal.lane:
            lane_guidance = "You are right of the goal. Prioritize action 0 (change left) to reach your destination. Avoid changing lanes in the opposite direction (action 2)"
        elif ego_lane < goal.lane:
            lane_guidance = "You are left of the goal. Prioritize action 2 (change right) to reach your destination. Avoid changing lanes in the opposite direction (action 0)."
        else:
            lane_guidance = "You are in the goal lane. Maintain lane and focus on speed."

        return f"""You are a highway driving agent. Your objective is to safely reach the waypoint.

    Priorities:
    1. Never collide.
    2. Reach the waypoint.
    3. Maintain speed within .1 of the speed limit by using brake or acceleration
    4. Avoid a front risk by using brake or change lane

    Current state:
    - Your speed: {ego_speed:.2f} (speed limit = {speed_limit:.2f})
    - Car ahead gap: {front_desc}
    - Left lane clear: {"yes" if left_clear else "no"}
    - Right lane clear: {"yes" if right_clear else "no"}
    - Your current lane: {ego_lane} of {num_lanes} lanes (0=leftmost, {num_lanes-1}=rightmost)
    {boundary_note}

    Waypoint:
    You have {distance:.1f} meters to reach lane {goal.lane}.
    {lane_guidance}

    Actions:
    0 = change left
    1 = keep lane
    2 = change right
    3 = accelerate
    4 = brake

    Situations (pick the one that best describes why you're choosing this action):
    passing_slow_vehicle = there's a slower vehicle ahead you're moving around
    yielding_to_traffic = another vehicle's position/speed requires you to hold back
    approaching_exit = you are converging on the waypoint's lane
    clearing_hazard = an unsafe situation requires an immediate correction
    maintaining_speed = no lane or speed correction is currently needed
    correcting_lane_error = you are off the goal lane and moving toward it

    Reply in exactly this format and nothing else:
    ACTION: <digit 0-4>
    SITUATION: <one situation name from the list above>"""
    
    SITUATIONS = [
        "passing_slow_vehicle",
        "yielding_to_traffic",
        "approaching_exit",
        "clearing_hazard",
        "maintaining_speed",
        "correcting_lane_error",
    ]

    def act(self, obs, goal, distance, lane_error, ego_lane, speed_limit, ego_speed, num_lanes):
        prompt = self.format_obs(obs, goal, distance, lane_error, ego_lane, speed_limit, ego_speed, num_lanes)
        try:
            response = requests.post(self.url, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.5}
            })
            text = response.json()["response"].strip()
            action = self._parse_action(text)
            situation = self._parse_situation(text)
            return action, situation
        except Exception as e:
            print(f"LLM call failed: {e}")
            return 1, "maintaining_speed"

    def _parse_action(self, text):
        match = re.search(r"ACTION:\s*([0-4])", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        # fallback: any lone digit 0-4 in the text
        match = re.search(r"[0-4](?!.*[0-4])", text)
        if match:
            return int(match.group())
        return 1  # fallback keep lane

    def _parse_situation(self, text):
        match = re.search(r"SITUATION:\s*(\w+)", text, re.IGNORECASE)
        if match:
            candidate = match.group(1).lower()
            if candidate in self.SITUATIONS:
                return candidate
        # fallback: check if any known situation word appears anywhere
        lowered = text.lower()
        for s in self.SITUATIONS:
            if s in lowered:
                return s
        return "maintaining_speed"



def fresh_facts():
    return {
        # --- safety/physical facts (existing) ---
        "ego_speed": None,
        "front_gap": None,
        "front_rel_speed": None,
        "left_gap": None,
        "right_gap": None,
        "left_rel_speed": None,
        "right_rel_speed": None,
        "speed_limit": SPEED_LIMIT,
        "ttc": None,
        "speed_slow": None,
        "speed_fast": None,
        "left_feasible": None,
        "right_feasible": None,
        "front_risk": None,
        "ego_lane": None,
        "num_lanes": None,

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

CONSTRAINT_TIERS = [
    ("safety",         ["avoid_collision"]),
    ("legality",       ["stay_legal"]),
    ("route",          ["preserve_route", "hold_route"]),
    ("no_regression",  ["no_lane_regression"]),   # new
    ("progress",       ["increase_progress"]),
    ("comfort",        ["maintain_comfort"]),
    ("brake_fallback", ["brake_for_front_risk"]),
    ("llm_preference", ["matches_llm_intent"]),
]

# situation label -> constraints that situation implies.
# This is the piece that grows independently of the fixed 5-action space.
SITUATION_CONSTRAINTS = {
    "passing_slow_vehicle":   {"preserve_route", "increase_progress"},
    "yielding_to_traffic":    {"maintain_comfort"},
    "approaching_exit":       {"preserve_route"},
    "clearing_hazard":        {"avoid_collision", "maintain_comfort"},
    "maintaining_speed":      {"increase_progress"},
    "correcting_lane_error":  {"preserve_route"},
}

TIER_NAMES = [name for name, _ in CONSTRAINT_TIERS]

def log_result(row: dict, path="results.csv"):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def log_tier_wins(mode, seed, mask_prob, tier_win_counts, path="tier_wins.csv"):
    row = {"mode": mode, "seed": seed, "mask_prob": mask_prob}
    for name, count in zip(TIER_NAMES, tier_win_counts):
        row[f"tier_{name}"] = count
    log_result(row, path=path)


def infer_constraints(llm_action, facts, situation=None):
    base = {"avoid_collision", "stay_legal", "maintain_comfort", "no_lane_regression"}

    if facts.get("front_risk") is True:
        base.add("brake_for_front_risk")

    # situation-derived: the LLM's semantic read of the moment.
    # This is open-ended vocabulary, decoupled from the 5-value action space.
    base |= SITUATION_CONSTRAINTS.get(situation, set())

    # fact-derived: active regardless of what the LLM proposed
    lane_error = facts.get("lane_error")
    if lane_error is not None and lane_error != 0:
        base.add("preserve_route")
    elif lane_error == 0:
        base.add("hold_route")  # already at goal lane — don't leave it

    if facts.get("speed_slow") is True:
        base.add("increase_progress")

    base.add("matches_llm_intent")
    return base

def satisfies_avoid_collision(predicted, facts):
    if predicted["front_risk"] is None:
        return None
    return not predicted["front_risk"]

def satisfies_brake_for_front_risk(action, facts):
    if facts["front_risk"] is not True:
        return None
    return action == 4

def satisfies_stay_legal(predicted, facts):
    if predicted["speed_fast"] is None:
        return None
    return not predicted["speed_fast"]

def satisfies_increase_progress(predicted, facts):
    if predicted["speed_slow"] is None:
        return None
    return not predicted["speed_slow"]

def satisfies_preserve_route(predicted, facts):
    lane_error = facts.get("lane_error")
    predicted_lane_error = predicted.get("lane_error")
    if lane_error is None or predicted_lane_error is None:
        return None
    if predicted_lane_error == 0:
        return True
    return abs(predicted_lane_error) < abs(lane_error)

def satisfies_hold_route(predicted, facts):
    predicted_lane_error = predicted.get("lane_error")
    if predicted_lane_error is None:
        return None
    return predicted_lane_error == 0

def satisfies_maintain_comfort(predicted, facts):
    if predicted["ego_speed"] is None or facts["ego_speed"] is None:
        return None
    if predicted["front_risk"]:
        return True  # abrupt action is justified under risk, not "uncomfortable"
    speed_delta = abs(predicted["ego_speed"] - facts["ego_speed"])
    return speed_delta <= DELTA_SPEED + 1e-6

def satisfies_matches_llm_intent(action, llm_action):
    return action == llm_action

def satisfies_no_lane_regression(predicted, facts):
    lane_error = facts.get("lane_error")
    predicted_lane_error = predicted.get("lane_error")
    if lane_error is None or predicted_lane_error is None:
        return None
    return abs(predicted_lane_error) <= abs(lane_error)

CONSTRAINT_CHECKS = {
    "avoid_collision": satisfies_avoid_collision,
    "stay_legal": satisfies_stay_legal,
    "increase_progress": satisfies_increase_progress,
    "preserve_route": satisfies_preserve_route,
    "hold_route": satisfies_hold_route,
    "maintain_comfort": satisfies_maintain_comfort,
    "brake_for_front_risk": satisfies_brake_for_front_risk,
    "no_lane_regression": satisfies_no_lane_regression,
}

def evaluate_action(action, facts, llm_action, intended):
    predicted = predict_effect(action, facts)
    scores = []
    results = {}
    for _, names in CONSTRAINT_TIERS:
        satisfied = 0
        for c in names:
            if c not in intended:
                continue
            if c == "matches_llm_intent":
                r = satisfies_matches_llm_intent(action, llm_action)
            else:
                r = CONSTRAINT_CHECKS[c](predicted, facts)
            results[c] = r
            if r is True:
                satisfied += 1
            elif r is None:
                satisfied += 0.5
        scores.append(satisfied)
    return tuple(scores), results


def select_action(facts, llm_action, situation=None):
    intended = infer_constraints(llm_action, facts, situation)
    feasible = feasible_actions(facts)

    print("FEASIBLE ACTIONS:", sorted(feasible))
    print("FACTS:", {
        "left_gap": facts["left_gap"],
        "right_gap": facts["right_gap"],
        "left_feasible": facts["left_feasible"],
        "right_feasible": facts["right_feasible"],
        "front_risk": facts["front_risk"],
        "speed_slow": facts["speed_slow"],
        "speed_fast": facts["speed_fast"],
        "lane_error": facts["lane_error"]
    })

    llm_score, llm_results = evaluate_action(llm_action, facts, llm_action, intended) \
        if llm_action in feasible else (None, {})

    best_action, best_score, best_results = None, None, None
    print("\nACTION DEBUG")
    print("lane_error:", facts["lane_error"])
    print("intended:", intended)

    for action in sorted(feasible):
        score, results = evaluate_action(action, facts, llm_action, intended)
        predicted = predict_effect(action, facts)

        print("\nAction:", action)
        print("score:", score)
        print("results:", results)
        print("predicted lane_error:", predicted.get("lane_error"))
        print("predicted speed:", predicted.get("ego_speed"))
        print("predicted front_risk:", predicted.get("front_risk"))
        if best_score is None or score > best_score:
            best_score, best_action, best_results = score, action, results

    if best_action is None:
        hard_feasible = hard_feasible_actions(facts)
        for action in sorted(hard_feasible) if hard_feasible else [4]:
            score, results = evaluate_action(action, facts, llm_action, intended)
            if best_score is None or score > best_score:
                best_score, best_action, best_results = score, action, results
        if best_action is None:
            best_action = 4  # brake is always physically valid as last resort
            best_score, best_results = (0,) * len(CONSTRAINT_TIERS), {}

    explanation = {
        "llm_action": llm_action,
        "llm_feasible": llm_action in feasible,
        "llm_results": llm_results,
        "chosen_action": best_action,
        "chosen_results": best_results,
        "overridden": best_action != llm_action,
    }

    

    return best_action, best_score, intended, explanation


def update_facts_from_obs(obs, facts, mask_prob=0.0):
    obs = np.array(obs).flatten().reshape(5, 5)  # 5 ve hicles x [presence,x,y,vx,vy]
    ego = obs[0]
    others = obs[1:]

    ego_speed = float(ego[3])

    front_gap, front_rel_speed = None, None
    left_gap, right_gap = None, None
    left_rel_speed, right_rel_speed = None, None

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
                left_rel_speed = float(ego[3] - vx)
        elif y > LANE_EPS:
            if right_gap is None or abs(x) < right_gap:
                right_gap = float(abs(x))
                right_rel_speed = float(ego[3] - vx)

    candidates = {
        "ego_speed": ego_speed,
        "front_gap": front_gap,
        "front_rel_speed": front_rel_speed,
        "left_gap": left_gap,
        "right_gap": right_gap,
        "left_rel_speed": left_rel_speed,
        "right_rel_speed": right_rel_speed,
    }

    for key, value in candidates.items():
        if value is not None and np.random.rand() >= mask_prob:
            facts[key] = value
        else:
            facts[key] = None



def prop_ttc(facts):
    gap = facts["front_gap"]
    rel_speed = facts["front_rel_speed"]
    if gap is None:
        # no vehicle detected in this lane at all — genuinely empty, not unknown
        facts["ttc"] = float("inf")
        return
    if rel_speed is None:
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
    facts["front_risk"] = ttc < .3


def prop_speed_slow(facts):
    speed = facts["ego_speed"]
    if speed is None:
        facts["speed_slow"] = None
        return
    facts["speed_slow"] = speed < facts["speed_limit"] - 10

def prop_speed_fast(facts):
    speed = facts["ego_speed"]
    if speed is None:
        facts["speed_fast"] = None
        return
    facts["speed_fast"] = speed > facts["speed_limit"]


def prop_left_feasible(facts):
    if facts.get("ego_lane") is not None and facts.get("num_lanes") is not None:
        if facts["ego_lane"] <= 0:
            facts["left_feasible"] = False
            return
    gap = facts["left_gap"]
    if gap is None:
        facts["left_feasible"] = None
        return
    facts["left_feasible"] = gap >= .09


def prop_right_feasible(facts):
    if facts.get("ego_lane") is not None and facts.get("num_lanes") is not None:
        if facts["ego_lane"] >= facts["num_lanes"] - 1:
            facts["right_feasible"] = False
            return
    gap = facts["right_gap"]
    if gap is None:
        facts["right_feasible"] = None
        return
    facts["right_feasible"] = gap >= .09


PROPAGATORS = [prop_ttc, prop_front_risk, prop_speed_slow, prop_speed_fast, prop_left_feasible, prop_right_feasible]


def run_propagators(facts):
    for p in PROPAGATORS:
        p(facts)


DELTA_SPEED = 5  # real speed units — matches the step size between TARGET_SPEEDS entries

def predict_effect(action, facts):
    predicted = dict(facts)

    if action == 3:
        if facts["ego_speed"] is not None:
            predicted["ego_speed"] = facts["ego_speed"] + DELTA_SPEED
    elif action == 4:
        if facts["ego_speed"] is not None:
            predicted["ego_speed"] = facts["ego_speed"] - DELTA_SPEED
        if facts["front_rel_speed"] is not None:
            predicted["front_rel_speed"] = facts["front_rel_speed"] - DELTA_SPEED
    elif action == 0:  # change left
        predicted["front_gap"] = None
        predicted["front_rel_speed"] = None
        if facts.get("lane_error") is not None:
            predicted["lane_error"] = facts["lane_error"] + 1
    elif action == 2:  # change right
        predicted["front_gap"] = None
        predicted["front_rel_speed"] = None
        if facts.get("lane_error") is not None:
            predicted["lane_error"] = facts["lane_error"] - 1
    # action == 1: no change

    run_propagators(predicted)
    return predicted


def feasible_actions(facts):
    feasible = {0, 1, 2, 3, 4}  # left, keep, right, accel, brake

    if facts["left_feasible"] is False:
        feasible.discard(0)
    if facts["right_feasible"] is False:
        feasible.discard(2)
    if facts["speed_fast"] is True:
        feasible.discard(1)
        feasible.discard(3)

    if facts["front_risk"] is True:
        for action in list(feasible):
            if action == 4:
                continue
            predicted = predict_effect(action, facts)
            if predicted["front_risk"] is True:
                feasible.discard(action)
        feasible.add(4)  # brake must always remain available as an escape hatch

    return feasible

def hard_feasible_actions(facts):
    """Physical/structural infeasibility only — never negotiable."""
    feasible = {0, 1, 2, 3, 4}
    if facts["left_feasible"] is False:
        feasible.discard(0)
    if facts["right_feasible"] is False:
        feasible.discard(2)
    return feasible


def known_fraction(facts):
    keys = ["ego_speed", "front_gap", "front_rel_speed", "left_gap", "right_gap"]
    known = sum(1 for k in keys if facts[k] is not None)
    return known / len(keys)



def run(seed, mask_prob=0.0, mode="intent", target_rejection_rate=None, num_steps=3):
    set_seed(seed)


    env = gym.make('highway-v0', 
        render_mode = "human",
        config={
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": [0, 5, 10, 15, 20, 25, 30, 35],
        },
        "vehicles_density" : 1,
        "lanes_count": 4,
    })
    num_lanes = env.unwrapped.config["lanes_count"]

    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    ego_start_x = env.unwrapped.vehicle.position[0]

    waypoints = [
        Waypoint(ego_start_x + 200, 0), 
        Waypoint(ego_start_x + 400, 3), 
        Waypoint(ego_start_x + 600, 0),  
    ]

    current_waypoint = 0


    policy = LLMPolicy()

    steps = 0
    crashes = 0
    rejected = 0
    action_counts = np.zeros(5)
    speed_sum = 0
    known_frac_sum = 0
    goal_abandoned = 0
    waypoints_reached = 0

    override_reasons = {}   # constraint_name -> count of times it was the failure
    tier_win_counts = np.zeros(len(CONSTRAINT_TIERS))  # which tier had first nonzero gap when overridden
    infeasible_overrides = 0  # overrides where the LLM action wasn't even in feasible_actions



    for _ in range(num_steps): #num_steps

        ego = env.unwrapped.vehicle
        ego_x = ego.position[0]
        ego_lane = ego.lane_index[2]

        goal = waypoints[current_waypoint]
        distance = abs(goal.x - ego_x)
        if ego_x >= goal.x or distance <= 30:
            if current_waypoint < len(waypoints) - 1:
                if ego_lane == waypoints[current_waypoint].lane:
                    waypoints_reached += 1
                current_waypoint += 1
                print(f"-- Waypoint updated to {current_waypoint} ---")
                goal = waypoints[current_waypoint]
                distance = abs(goal.x - ego_x)

        lane_error = goal.lane - ego_lane
        llmaction, situation = policy.act(obs, goal, distance, lane_error, ego_lane, SPEED_LIMIT, ego.speed, num_lanes)
        action_counts[llmaction] += 1

        pre_step_speed = ego.speed
        facts = fresh_facts()
        update_facts_from_obs(obs, facts, mask_prob=mask_prob)
        facts["ego_speed"] = ego.speed  # overwrite normalized value with real units
        facts["goal_distance"] = distance
        facts["goal_lane"] = goal.lane
        facts["lane_error"] = lane_error
        facts["ego_lane"] = ego_lane
        facts["num_lanes"] = num_lanes  
        run_propagators(facts)
        print(f"  left_gap={facts['left_gap']} left_feasible={facts['left_feasible']} right_gap={facts['right_gap']} right_feasible={facts['right_feasible']} front_risk={facts['front_risk']}")
        facts["goal_distance"] = distance
        facts["goal_lane"] = goal.lane
        facts["lane_error"] = lane_error
        known_frac_sum += known_fraction(facts)

        explanation = None
        constraint_score = None
        intended_constraints = None

        if mode == "raw": #pure llm
            action = llmaction
            overridden = False

        elif mode == "shield": #brake fallback
            allowed = feasible_actions(facts)
            if llmaction not in allowed:
                action = 4
                overridden = True
            else:
                action = llmaction
                overridden = False

        elif mode == "random_matched": #llm chooses infeasible randomly choose new action
            allowed = feasible_actions(facts)
            rate = target_rejection_rate if target_rejection_rate is not None else 0.0
            if llmaction not in allowed or np.random.rand() < rate:
                action = random.choice(list(allowed)) if allowed else 1
                overridden = True
            else:
                action = llmaction
                overridden = False

        else:  # "intent"
            final_action, constraint_score, intended_constraints, explanation = select_action(facts, llmaction, situation)
            action = final_action
            overridden = explanation["overridden"]

        # unified goal_preserved definition across all modes
        predicted_for_chosen = predict_effect(action, facts)
        predicted_lane_error = predicted_for_chosen.get("lane_error")
        if predicted_lane_error is None or lane_error is None:
            goal_preserved = None
        else:
            goal_preserved = abs(predicted_lane_error) <= abs(lane_error)

        if overridden:
            rejected += 1
            if goal_preserved is False:
                goal_abandoned += 1

        obs, reward, terminated, truncated, info = env.step(action)

        speed = obs[0][3] if hasattr(obs, "__len__") else 0
        speed_sum += speed
        steps += 1

        if isinstance(info, dict) and info.get("crashed", False):
            crashes += 1
            print("\n\n! CRASH DETECTED !!!")
            print("action:", action)
            print("LLM action:", llmaction)
            print("situation:", situation)
            print("facts:", facts)
            print("ego speed:", ego.speed)
            print("lane:", ego_lane)

        if terminated or truncated:
            obs, info = env.reset(seed=seed)
            ego_start_x = env.unwrapped.vehicle.position[0]
            waypoints = [
                Waypoint(ego_start_x + 200, 0),
                Waypoint(ego_start_x + 400, 3),
                Waypoint(ego_start_x + 600, 0),
            ]
            current_waypoint = 0

        print(f"\nego location : {ego_x:.1f} | goal_lane: {goal.lane} | curr lane: {ego_lane} | distance : {distance}")
        print(f"LLM proposed: {llmaction} ({situation}) | final action: {action} | overridden: {overridden}")
        print(f"\n Speed at decision time: {pre_step_speed:.3f} | Speed after action: {env.unwrapped.vehicle.speed:.3f}")

        if mode == "intent": #hierarchical constraints
            if explanation["overridden"]:
                if not explanation["llm_feasible"]:
                    infeasible_overrides += 1
                    print(f"  LLM action {llmaction} was infeasible (excluded before scoring)")
                else:
                    failed = [c for c, r in explanation["llm_results"].items() if r is False]
                    gained = [c for c, r in explanation["chosen_results"].items() if r is True]
                    print(f"  LLM action failed: {failed} | replacement satisfies: {gained}")

                    for c in failed:
                        override_reasons[c] = override_reasons.get(c, 0) + 1

                    llm_score_vec, _ = evaluate_action(llmaction, facts, llmaction, intended_constraints)
                    for i, (a, b) in enumerate(zip(llm_score_vec, constraint_score)):
                        if a != b:
                            tier_win_counts[i] += 1
                            break

            print(f"  score: {constraint_score} | intended: {intended_constraints}")

    final_ego_x = env.unwrapped.vehicle.position[0]
    final_goal = waypoints[current_waypoint]
    final_distance_to_goal = abs(final_goal.x - final_ego_x)
    route_progress = current_waypoint + max(0, 1 - final_distance_to_goal / 200)

    print(f"\noverride reasons (constraint -> times it failed on LLM action):", override_reasons)
    print("tier that decided each override:", dict(zip([t for t,_ in CONSTRAINT_TIERS], tier_win_counts)))
    print("infeasible-triggered overrides:", infeasible_overrides)

    print(f"\noverride reasons (constraint -> times it failed on LLM action):", override_reasons)
    print("tier that decided each override:", dict(zip([t for t,_ in CONSTRAINT_TIERS], tier_win_counts)))
    print("infeasible-triggered overrides:", infeasible_overrides)

    return (
                crashes / max(1, steps),
                crashes,
                waypoints_reached,
                current_waypoint,
                final_distance_to_goal,
                route_progress,
                action_counts / action_counts.sum(),
                speed_sum / max(1, steps),
                rejected / max(1, steps),
                rejected,
                known_frac_sum / max(1, steps),
                goal_abandoned,
                override_reasons,
                tier_win_counts,
                infeasible_overrides,
            )




if __name__ == "__main__":

    seeds = list(range(2))

    # set MASK_PROB > 0 to simulate sensor dropout and test graceful degradation
    MASK_PROB = 0

    crash_rates = []
    crashes_list = []
    action_distributions = []
    speeds = []
    rejected_rates = []
    rejected_list = []
    known_fracs = []
    goal_abandoned_list = []
    combined_override_reasons = {}
    combined_tier_wins = np.zeros(len(CONSTRAINT_TIERS))
    total_infeasible_overrides = 0

    print("running propagator network ...")
    print(f"mask probability: {MASK_PROB}")

    for s in seeds:
        (crash_rate, crashes, action_dist, speed, rejected_rate, rejected,
         known_frac, goal_abandoned, override_reasons, tier_win_counts,
         infeasible_overrides) = run(
            s, mask_prob=MASK_PROB, mode="intent", num_steps = 30
        )

        crash_rates.append(crash_rate)
        crashes_list.append(crashes)
        action_distributions.append(action_dist)
        speeds.append(speed)
        rejected_rates.append(rejected_rate)
        rejected_list.append(rejected)
        known_fracs.append(known_frac)
        goal_abandoned_list.append(goal_abandoned)
        combined_tier_wins += tier_win_counts
        total_infeasible_overrides += infeasible_overrides
        for c, n in override_reasons.items():
            combined_override_reasons[c] = combined_override_reasons.get(c, 0) + n

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

    print("\nGoal preservation:")
    print("total goal-abandoned overrides:", np.sum(goal_abandoned_list))

    print("\nConstraint diagnostics:")
    print("override reasons (constraint -> times it failed on LLM action):", combined_override_reasons)
    print("tier that decided each override:", dict(zip([t for t, _ in CONSTRAINT_TIERS], combined_tier_wins)))
    print("total infeasible-triggered overrides:", total_infeasible_overrides)

    