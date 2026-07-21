import gymnasium as gym
import highway_env

env = gym.make(
    'highway-v0',
    render_mode="human",
    config={
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": [0, 5, 10, 15, 20, 25, 30, 35],
        },
        "vehicles_density": 1,
        "lanes_count": 4,
    },
)

obs, info = env.reset(seed=0)
ego = env.unwrapped.vehicle

print(f"start lane: {ego.lane_index[2]}")

# Force lane changes: right, right, right (should go 0->1->2->3 with pauses to let it settle)
actions = [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]

for i, a in enumerate(actions):
    obs, reward, terminated, truncated, info = env.step(a)
    ego = env.unwrapped.vehicle
    print(f"step {i}: action={a} | lane={ego.lane_index[2]} | x={ego.position[0]:.1f} | speed={ego.speed:.2f}")
    if terminated or truncated:
        obs, info = env.reset(seed=0)