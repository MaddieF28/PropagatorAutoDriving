import gymnasium as gym
import highway_env

env = gym.make("highway-v0")
env.reset()
print(type(env.unwrapped.vehicle))
print(env.unwrapped.vehicle.__class__.__mro__)