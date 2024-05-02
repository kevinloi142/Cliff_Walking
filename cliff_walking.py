import gym
import numpy as np
import matplotlib.pyplot as plt

def run(episodes, render=False) :

    env = gym.make("CliffWalking-v0", render_mode = 'human' if render else None)

    q = np.zeros((env.observation_space.n, env.action_space.n))

    learning_rate = 0.9
    discount_factor = 0.9 

    epsilon_decay_rate = 0.01
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes) #keep track of episodes
    
    for i in range(episodes):
        is_done = False #True when goal is reached or the guy falls into the lake
        truncated = False #When too many actions are taken

        
        state = env.reset()[0]

        while(not is_done and not truncated):
            if rng.random() < epsilon_decay_rate:
                action = env.action_space.sample() #actions: 0=right, 1=down, 2=right, 3=up
            else :
                action = np.argmax(q[state,:])

            new_state, reward, is_done, truncated, _ = env.step(action)

            #Q equation
            q[state, action] = q[state, action] + learning_rate * (reward + discount_factor * np.max(q[new_state]) - q[state, action])

            state = new_state

            if(is_done or rewards_per_episode[i] < -100) :
                truncated = True
            else:
                rewards_per_episode[i] -= 1
        
        
   
    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        #Plot every 10 episodes
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-10):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('cliffwalking.png')


if __name__ == '__main__' :
    #run(3)
    run(150)
