import gymnasium as gym
import gym_super_mario_bros

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from nes_py.wrappers import JoypadSpace

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.SuperMarioBrosEnv('vanilla')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, progress_bar=True)
model.save("m")
del model 

model = PPO.load("m")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"mean_reward: {mean_reward} +- {std_reward}")

done = True
for step in range(5000):
    if done:
        state = env.reset()
        state, reward, done, info = env.step(model.predict(state))
        env.render()

env.close()

# observation, info = env.reset()

# for _ in range(200):
#     action = env.action_space.sample()  # agent policy that uses the observation and info
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()

# env.close()
