import gymnasium as gym
# import gym
import gym_examples
from stable_baselines3 import DQN
import matplotlib.pyplot as plt 

# Create a Gym environment
env = gym.make("gym_examples/CountersGame-v0",render_mode="rgb_array")

# Define the Q-learning agent
model = DQN("MultiInputPolicy", env, verbose=1)
# model = DQN("MlpPolicy", env, verbose=1)

# Train the agent
# model.learn(total_timesteps=60000,progress_bar=True)
model.learn(total_timesteps=60000)

# Save the trained agent
model.save("q_learning_agent")

# Evaluate the trained agent
total_reward = 0
num_episodes = 30

counters_win=[]
win=[]
reward_win=[]
for j in range(num_episodes):
    print(j)
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        observation, reward, done, truncated, info = env.step(action)
        counters_win.append(observation['counters'])
    if done and truncated==False:
        win.append(counters_win)
        reward_win.append(reward)
    if done or truncated:
        counters_win=[]

total_reward += reward
average_reward = total_reward / num_episodes
print("Average reward:", average_reward)   
ix=[sum(l) for l in win].index(min([sum(l) for l in win]))
plt.figure(figsize=(24,8))
plt.bar([str(x) for x in range(len(win[ix]))], win[ix], color ='maroon',width = 0.4)
plt.show()   
print(reward_win)  
        
# observation, info = env.reset()
# # observation = env.reset()
# counters_win=[]
# win=[]
# frames = []
# for _ in range(112000):
#     action = model.predict(observation)  # agent policy that uses the observation and info
#     observation, reward, terminated, truncated, info = env.step(int(action[0]))
#     counters_win.append(observation['counters'])
#     print(_,observation,'truncated',truncated,'terminated',terminated,'reward',reward)
#     frames.append(env.render())
#     if terminated and truncated==False:
#         win.append(counters_win)
#         # plt.figure(figsize=(24,8))
#         # plt.bar([str(x) for x in range(len(counters_win))], counters_win, color ='maroon',width = 0.4)
#         # plt.show()
        
#     if terminated or truncated:
#         observation, info = env.reset()
#         counters_win=[]
        
# ix=[sum(l) for l in win].index(min([sum(l) for l in win]))
# plt.figure(figsize=(24,8))
# plt.bar([str(x) for x in range(len(win[ix]))], win[ix], color ='maroon',width = 0.4)
# plt.show()
        





# gym.pprint_registry()
# pygame.quit()



