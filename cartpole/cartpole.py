# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:18:08 2020

@author: daniel arnold
"""

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

env_name = "CartPole-v1"

class DQNSolver:
    
    def __init__(self, observation_space, action_space, epsilon_decay):
        
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.MEMORY_SIZE = 2000
        self.BATCH_SIZE = 64
        
        self.action_space = action_space
        #deque: list-like container with fast appends and pops on either end
        self.memory = deque(maxlen=self.MEMORY_SIZE)
        
        #NN parameters
        #1 hidden layer NN
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        
    def remember(self, state, action, reward, next_state, done):
        #store experience in a list
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        #act method for training
        #exploration
        #check to see if the exploration rate is greater than a random action
        #as the exploration rate decays, we will take random actions less often
        if(np.random.rand() < self.epsilon):
            #randomly choose an action between 0 and 1
            return random.randrange(self.action_space)
        #predict q_values from present state
        q_vals = self.model.predict(state)
        #choose best action and return
        return np.argmax(q_vals[0])
    
    def act_test(self, state):
        #act method for testing
        #predict q_values from present state
        q_vals = self.model.predict(state)
        #choose best action and return
        return np.argmax(q_vals[0])
        
    def experience_replay(self):
        if(len(self.memory) < self.BATCH_SIZE):
            #not enough experiences for replay
            return
        
        #get a random sample from the memory buffer
        batch = random.sample(self.memory, self.BATCH_SIZE)
        #do replay
        for state, action, reward, next_state, done in batch:
            q_update = reward
            if not done:
                #q learning
                q_update = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            
            #get the q_values from the present NN
            q_vals = self.model.predict(state)
            #update q_vals with new q_vals from specific actions taken
            q_vals[0][action] = q_update
            #fit the model
            self.model.fit(state, q_vals, verbose=0)
            
            #decrease the exploration
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
    def get_epsilon(self):
        return self.epsilon
    
def plot_results(results, mean_results, episode, num_ticks_to_win):
    plt.clf()
    plt.plot(results,'b', label = "score")
    plt.plot(mean_results,'r',label="mean score")
    tick_plot = np.arange(0,episode)
    plt.plot(tick_plot, np.ones(np.shape(tick_plot)) * num_ticks_to_win,'k-', label="win threshold")
    plt.show()
    plt.xlabel('Episode Number')
    plt.ylabel('Episode Score')
    plt.title('Cartpole scores')
    plt.legend()

def cartpole(num_episodes, epsilon_decay, verbose = True, make_plots = True):
    
    if verbose:
        print("Solving CartPole with DQN using Keras and TensorFlow")
    
    scores = deque(maxlen=100)
    score_list = []
    mean_score_list = []
    num_trials = -1
    
    num_ticks_to_win = 195
    
    #training environment
    env = gym.make(env_name)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    
    #testing environment
    env_test = gym.make(env_name)
    
    #initalize the DQN solver
    dqn_solver = DQNSolver(observation_space, action_space, epsilon_decay)
    
    #run forever until we have a DQN that generates a policy that achieves the
    #objectives of the environment
    
    for episode in range(0, num_episodes):
        #training
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        done = False
        step = 0
        while not done:
            step += 1
            #env.render()
            action = dqn_solver.act(state)
            next_state, reward, done, info = env.step(action)
            #every frame robot is balanced, earn +1, else earn -1
            #reward = reward if not done else -reward
            next_state = np.reshape(next_state, [1, observation_space])
            dqn_solver.remember(state, action, reward, next_state, done)
            state = next_state
            
        #testing
        state_test = env_test.reset()
        state_test = np.reshape(state_test, [1, observation_space])
        done_test = False
        step = 0
        while not done_test:
            step += 1
            #env.render()
            action = dqn_solver.act_test(state_test)
            next_state_test, reward_test, done_test, info_test = env_test.step(action)
            #every frame robot is balanced, earn +1, else earn -1
            #reward = reward if not done else -reward
            next_state_test = np.reshape(next_state_test, [1, observation_space])
            #dqn_solver.remember(state, action, reward, next_state, done)
            state_test = next_state_test
        
        #number of steps the pole was upright
        scores.append(step)
        mean_score = np.mean(scores)
        
        score_list.append(step)
        mean_score_list.append(mean_score)
        
        if mean_score >= num_ticks_to_win and episode >= 100:
            if verbose:
                print('Ran {} episodes. Solved after {} trials'.format(episode, episode - 100))
            num_trials = episode - 100
            break
        if verbose and episode % 100 == 0:
            print('[Episode {}] - Mean survival time over last 100 episodes was {}.'.format(episode, mean_score))
            

        dqn_solver.experience_replay()
        
        if verbose:
            print("episode", episode, "epsilon:", dqn_solver.get_epsilon(), "score:", step)
    
    if make_plots:
        plot_results(score_list, mean_score_list, episode, num_ticks_to_win)
    
    if verbose:
        print("Program Complete")
        
    return score_list, mean_score_list, num_trials
            
if __name__ == "__main__":
    cartpole(1000, 0.9995)