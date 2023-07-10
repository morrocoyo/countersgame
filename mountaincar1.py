import gymnasium as gym
# import gym
import matplotlib.pyplot as plt 
env = gym.make('MountainCar-v0',render_mode="human")

observation, info = env.reset()

for _ in range(200):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
