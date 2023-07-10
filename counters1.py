import gymnasium as gym
# import gym
import gym_examples
import matplotlib.pyplot as plt 
from matplotlib import animation
env = gym.make("gym_examples/CountersGame-v0",render_mode="rgb_array")
# env = gym.make("gym_examples/CountersGame-v0",render_mode="human")

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    # anim.save(path + filename, writer='imagemagick', fps=60)
    anim.save(path + filename, writer='imagemagick', fps=10)
    
# env = gym.make("gym_examples/CountersGame-v0")

observation, info = env.reset()
# observation = env.reset()
counters_win=[]
win=[]

frames = []
for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    counters_win.append(observation['counters'])
    print(_,observation,'truncated',truncated,'terminated',terminated,'reward',reward)
    # frames.append(env.render())
    if terminated and truncated==False:
        win.append(counters_win)
        # plt.figure(figsize=(24,8))
        # plt.bar([str(x) for x in range(len(counters_win))], counters_win, color ='maroon',width = 0.4)
        # plt.show()
        
    if terminated or truncated:
        observation, info = env.reset()
        counters_win=[]
        
ix=[sum(l) for l in win].index(min([sum(l) for l in win]))
plt.figure(figsize=(20,7))
plt.bar([str(x) for x in range(len(win[ix]))], win[ix], color ='maroon',width = 0.4)
plt.show()

import pickle
pickle.dump(win[ix],'Data/pasa nivel_7_jul')

env.close()
save_frames_as_gif(frames)

# gym.pprint_registry()
# pygame.quit()

import pandas as pd
distri_dia = pd.read_csv('gym-examples/sum_distri_dia.csv')
distri_dia.set_index('Unnamed: 0',inplace=True)
distri_dia = distri_dia.reindex(range(1440))
distri_dia = distri_dia.fillna(0)
plt.figure(figsize=(15,5))
plt.plot(distri_dia)
plt.show()

