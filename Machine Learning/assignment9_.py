import gymnasium as gym
import numpy as np
import random
from IPython.display import clear_output

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode='rgb_array')

# Initialize Q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 1 

# Q-Learning
total_epochs, total_penalties, total_reward = 0, 0, 0
episodes = 100001

for i in range(episodes):
    state = env.reset()[0]
    epochs, penalties, reward = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, info, prob = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state
        epochs += 1
        
        if epsilon > 0:
            epsilon -= epsilon/episodes

    if reward == 0:
            penalties += 1

    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

    total_epochs += epochs
    total_reward += reward
    total_penalties += penalties

print("Final Q-table: \n", q_table)
print(f"\nAverage Number of Timesteps per Episode: {total_epochs / episodes}")
print("\nNumber of Falls: {}".format(total_penalties))
print("\nNumber of Goals: {}".format(total_reward))
