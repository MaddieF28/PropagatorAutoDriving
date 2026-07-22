import numpy as np
from full_propagators import run, log_result, log_tier_wins
import pandas as pd
import os

def already_done(mode, mask_prob, seed, path="results.csv"):
    if not os.path.isfile(path):
        return False
    df = pd.read_csv(path)
    return ((df["mode"] == mode) & (df["mask_prob"] == mask_prob) & (df["seed"] == seed)).any()


SEEDS = range(5)
MASK_PROBS = [0, 0.25, 0.5]
MODES = ["intent"]
NUM_STEPS = 40

for mode in MODES:
    for mask_prob in MASK_PROBS:
        for seed in SEEDS:
            if already_done(mode, mask_prob, seed):
                print(f"skipping, already have: mode={mode} mask_prob={mask_prob} seed={seed}")
                continue
            (crash_rate, crashes, waypoints_reached, current_waypoint, final_distance_to_goal,
            route_progress, action_dist, speed, rejected_rate, rejected, known_frac,
            goal_abandoned, override_reasons, tier_win_counts, infeasible_overrides) = run(
                seed, mask_prob=mask_prob, mode=mode, num_steps=NUM_STEPS
            )

            row = {
            "mode": mode,
            "seed": seed,
            "mask_prob": mask_prob,
            "crash_rate": crash_rate,
            "crashes": crashes,
            "waypoints_reached": waypoints_reached,
            "current_waypoint": current_waypoint,
            "final_distance_to_goal": final_distance_to_goal,
            "route_progress": route_progress,
            "rejected_rate": rejected_rate,
            "goal_abandoned": goal_abandoned,
            "known_frac": known_frac,
            "avg_speed": speed,
        }
            for i, frac in enumerate(action_dist):
                row[f"action_dist_{i}"] = frac

            log_result(row, path="results.csv")
            log_tier_wins(mode, seed, mask_prob, tier_win_counts, path="tier_wins.csv")

            print(f"done: mode={mode} mask_prob={mask_prob} seed={seed}")