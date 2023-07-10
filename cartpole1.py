
#sudo apt-get install freeglut3-dev libgl1-mesa-dev libglu1-mesa-dev
import gymnasium as gym
# import gym
import matplotlib.pyplot as plt 

env = gym.make('CartPole-v1', render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

# save_frames_as_gif(frames)
env.close()






