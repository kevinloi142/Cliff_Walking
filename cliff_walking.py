import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, render=False, getSolution = False) :

    env = gym.make("CliffWalking-v0", render_mode = 'human' if render else None)

    if getSolution == False:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open("cliffwalking.pk0", 'rb') #grab the solution from cliffwalking.pk0.
        q = pickle.load(f)
        f.close()

    learning_rate = 0.8
    discount_factor = 0.8

    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes) #keep track of episodes
    
    for i in range(episodes):
        is_done = False #True when goal is reached or the guy falls into the lake
        truncated = False #When too many actions are taken

        
        state = env.reset()[0]

        while(not is_done or not truncated):
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
                rewards_per_episode[i] += reward
        
    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-10):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('cliffwalking.png')

    #save the q table so we can see the solution
    f = open("cliffwalking.pk0","wb")
    pickle.dump(q,f) #store q table into cliffwalking.pk0
    f.close()


if __name__ == '__main__' :
    #run(3)
    run(300)
    run(1, True, True)
