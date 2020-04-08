# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:18:08 2020

@author: daniel arnold
"""

import random
import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import datetime

#tensorflow inputs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

env_name = "CartPole-v1"

class DQNSolver:
    
    def __init__(self, observation_space, action_space, epsilon_decay, log_dir):
        
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.MEMORY_SIZE = 2000
        self.BATCH_SIZE = 200
        
        self.action_space = action_space
        #deque: list-like container with fast appends and pops on either end
        self.memory = deque(maxlen=self.MEMORY_SIZE)
        
        #writer for tensorboard
        self.summary_writer = tf.summary.FileWriter(log_dir)
        
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
    
    def act_greedy(self, state):
        #act method for testing - greedy policy
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
    
    def log_results(self, name, val, idx):
        #log results for visualization in Tensorboard
        summary = tf.summary.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = val
        summary_value.tag = name
        self.summary_writer.add_summary(summary, idx)
        self.summary_writer.flush()
            
###############################################################################
    
def plot_results(results, mean_results, episode, num_ticks_to_win):
    plt.clf()
    plt.plot(results,'b', label = "score")
    plt.plot(mean_results,'r',label="mean score")
    tick_plot = np.arange(0,episode)
    plt.plot(tick_plot, np.ones(np.shape(tick_plot)) * num_ticks_to_win,'k-', label="win threshold")
    plt.xlabel('Episode Number')
    plt.ylabel('Episode Score')
    plt.title('Cartpole scores')
    plt.legend()
    plt.show()

def cartpole(num_episodes, epsilon_decay, log_dir, verbose = True, make_plots = True):
    
    if verbose:
        print("Solving CartPole with DQN using Keras and TensorFlow")
    
    scores = deque(maxlen=100)
    score_list = []
    mean_score_list = []
    num_trials = -1
    
    num_ticks_to_win = 195
    
    #training environment - exploration policy
    env = gym.make(env_name)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    
    #testing environment - greedy policy
    env_greedy = gym.make(env_name)
    
    #initalize the DQN solver
    dqn_solver = DQNSolver(observation_space, action_space, epsilon_decay, log_dir)
    
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
            
        #test environment - greedy policy
        state_greedy = env_greedy.reset()
        state_greedy = np.reshape(state_greedy, [1, observation_space])
        done_greedy = False
        step = 0
        while not done_greedy:
            step += 1
            #env.render()
            action = dqn_solver.act_greedy(state_greedy)
            next_state_greedy, reward_greedy, done_greedy, info_greedy = env_greedy.step(action)
            #every frame robot is balanced, earn +1, else earn -1
            #reward = reward if not done else -reward
            next_state_greedy = np.reshape(next_state_greedy, [1, observation_space])
            #dqn_solver.remember(state, action, reward, next_state, done)
            state_greedy = next_state_greedy
        
        #number of steps the pole was upright
        scores.append(step)
        dqn_solver.log_results('scores', step, episode)
        
        mean_score = np.mean(scores)
        dqn_solver.log_results('mean_score(100)', mean_score, episode)
        
        score_list.append(step)
        mean_score_list.append(mean_score)
        
        if mean_score >= num_ticks_to_win and episode >= 100:
            if verbose:
                print('Ran {} episodes. Solved after {} trials'.format(episode, episode - 100))
            num_trials = episode - 100
            break
        if verbose and episode % 100 == 0:
            print('[Episode {}] - Mean survival time over last 100 episodes was {}.'.format(episode, mean_score))
        
        #experience replay - update Q approximation
        dqn_solver.experience_replay()
        
        if verbose:
            print("episode", episode, "epsilon:", dqn_solver.get_epsilon(), "score:", step)
    
    if make_plots:
        plot_results(score_list, mean_score_list, episode, num_ticks_to_win)
    
    if verbose:
        print("Program Complete")
        
    return score_list, mean_score_list, num_trials
            
if __name__ == "__main__":
    
    #log directory for TensorBoard
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #to activate Tensorboard:
    #tensorboard --logdir logs
    
    cartpole(1000, 0.9995, log_dir)