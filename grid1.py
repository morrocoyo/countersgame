import gymnasium as gym
# import gym
import gym_examples
# import matplotlib.pyplot as plt 
env = gym.make("gym_examples/GridWorld-v0",render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

# gym.pprint_registry()
