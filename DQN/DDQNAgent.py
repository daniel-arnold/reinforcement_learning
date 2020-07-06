# -*- coding: utf-8 -*-
"""
inspired by: http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
and https://rubikscode.net/2020/01/27/double-dqn-with-tensorflow-2-and-tf-agents-2/
"""

import logging
import numpy as np
#tensorflow inputs
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.losses import Huber
from collections import deque
import random

#####################################
########## Memory Buffer ############
#####################################

class ExperienceReplay:
    def __init__(self, maxlen):
        self._buffer = deque(maxlen=maxlen)
        
    def store(self, state, action, reward, next_state, done):
        self._buffer.append((state, action, reward, next_state, done))
        
    def get_batch(self, batch_size):
        sample_num = len(self._buffer)
        if(sample_num >  batch_size):
            return random.sample(self._buffer, sample_num)
        else:
            return random.sample(self._buffer, batch_size)
        
    def get_arrays_from_batch(self, batch, num_states):
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([(np.zeros(num_states) if x[4] is True else x[3]) 
                                for x in batch])
        dones = np.array([x[4] for x in batch])
        return states, actions, rewards, next_states, dones
   
    @property
    def buffer_size(self):
        return len(self._buffer)

#####################################
############## Agent ################
#####################################  

class DDQNAgent:
    def __init__(self, nn_params, hyp_params, observation_space, action_space):
        
        self.nn_params = nn_params

        self.epsilon_decay = hyp_params['epsilon_decay']
        self.epsilon_max = 1.0
        self.epsilon = self.epsilon_max
        self.epsilon_min = 0.01
        self.lr = hyp_params['lr']
        self.gamma = hyp_params['gamma']
        self.tau = hyp_params['tau']
        self.clipnorm_val = hyp_params['clipnorm_val']
        self.batch_size = hyp_params['batch_size']       
        self.memory_size = hyp_params['memory_size']
        self.action_space = action_space
        self.observation_space = observation_space
        self.er = ExperienceReplay(self.memory_size)
        
        logging.getLogger().setLevel(logging.INFO)
        
        self.Q_prime = self._build_network()
        if(self.clipnorm_val <= 0):
            self.Q_prime.compile(loss=Huber(), 
                                 optimizer=Adam(lr=self.lr))
        else:
            self.Q_prime.compile(loss=Huber(), 
                                 optimizer=Adam(lr=self.lr, 
                                                clipnorm=self.clipnorm_val))
        
        self.Q_target = self._build_network()
        self.Q_target.set_weights(self.Q_prime.get_weights())
        
    def _build_network(self):
        inputs = Input(shape=(self.observation_space, ), name='state')
        x = inputs
        for i in range(0,self.nn_params['num_layers']):
            x = Dense(self.nn_params['hidden_neurons'][i], 
                      activation=self.nn_params['activations'][i],
                      kernel_initializer=he_normal())(x)
        outputs = Dense(self.action_space, activation='linear')(x)
        net = Model(inputs, outputs)
        return net
                
    def update_target_network(self):
        weights = []
        for t,p in zip(self.Q_target.get_weights(), 
                       self.Q_prime.get_weights()):
            weights.append((1 - self.tau) * t + self.tau * p)
        
        self.Q_target.set_weights(weights)
        #self.Q_target.set_weights(self.Q_prime.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.er.store(state, action, reward, next_state, done)
        
    def get_action(self, state):
        #exploration
        #check to see if the exploration rate is greater than a random action
        #as the exploration rate decays, we will take random actions less often
        if(np.random.rand() < self.epsilon):
            #randomly choose an action between 0 and 1
            return random.randint(0, self.action_space-1)
        #predict q_values from present state
        q_vals = self.Q_prime.predict_on_batch(state[None,:])
        #choose best action and return
        return  np.argmax(q_vals)
    
    def get_action_greedy(self, state):
        #predict q_values from present state
        q_vals = self.Q_prime.predict_on_batch(state[None,:])
        #choose best action and return
        return np.argmax(q_vals)
    
    def train_network(self):
        if self.er.buffer_size < self.batch_size:
            return 0
        #get a batch from the memory buffer
        batch = self.er.get_batch(self.batch_size)
        states, actions, rewards, next_states, dones = \
            self.er.get_arrays_from_batch(batch, self.observation_space)
            
        #predict Q(s,a) and Q(s',a') of primary network
        q_vals = (self.Q_prime.predict_on_batch(states)).numpy()
        q_vals_next_state = self.Q_prime.predict_on_batch(next_states)
        #find the optimal actions for Q(s',a')
        actions_optimal = np.argmax(q_vals_next_state, axis=1)
        
        batch_idxs = np.arange(self.er.buffer_size)  
        #predict Q(s',a') for target network
        q_next_state_target = self.Q_target.predict_on_batch(next_states)
        #compute the update using Q_target with Q_primary optimal actions

        q_next_state_target = q_next_state_target.numpy()
        q_update = rewards + \
            self.gamma * q_next_state_target[batch_idxs, actions_optimal] * (1 - dones)
         
        #update q_vals for training and train
        q_vals[batch_idxs,actions] = q_update
        loss = self.Q_prime.train_on_batch(states, q_vals)
        
        #update target network (low pass filter)
        self.update_target_network()
        
        return loss

    def get_epsilon(self):
        return self.epsilon

    def train(self, env, batch_sz, updates):
        #training loop
        ep_rewards = []
        ep_losses = []
        rewards = []
        next_state = env.reset()
        for update in range(updates):
            for step in range(batch_sz):
                state = next_state
                action = self.get_action(next_state)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                rewards.append(reward)
                
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

                if done:
                    ep_rewards.append(sum(rewards))
                    rewards = []
                    next_state = env.reset()
                    
                    msg = "Episode: %03d, Reward: %03d, epsilon: %f, update: %03d"
                    fmt = (len(ep_rewards)-1, ep_rewards[-1], self.epsilon, update)
                    logging.info(msg % fmt)
            
            #sample the buffer and train the Qnetwork
            ep_loss = self.train_network()
            ep_losses.append(ep_loss)
            
            logging.debug("[%d/%d] value loss: %s" 
                          % (update + 1, updates, ep_loss))
        return ep_rewards, ep_losses
 
    def test(self, env, render=False):
        state, done, episode_reward = env.reset(), False, 0
        while not done:
            action = self.get_action_greedy(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
        return episode_reward