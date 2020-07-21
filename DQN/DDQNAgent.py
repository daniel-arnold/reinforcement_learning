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
import random
from collections import deque

#####################################
########## Memory Buffer ############
#####################################

class ExperienceReplay:
    def __init__(self, buffer_len):
        
        self.buffer = deque(maxlen=buffer_len)
        
    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def get_batch(self, memory_batch_size):
        sample_num = len(self.buffer)
        if(sample_num <  memory_batch_size):
            return random.sample(self.buffer, sample_num)
        else:
            return random.sample(self.buffer, memory_batch_size)
        
    def get_arrays_from_batch(self, batch, num_states):
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([(np.zeros(num_states) if x[4] is True else x[3]) 
                                for x in batch])
        dones = np.array([x[4] for x in batch])
        return states, actions, rewards, next_states, dones
   
    def buffer_size(self):
        return len(self.buffer)

#####################################
############## Agent ################
#####################################  

class DDQNAgent:
    def __init__(self, nn_params, hyp_params, observation_space, action_space):
        
        self.nn_params = nn_params

        self.epsilon_decay = hyp_params['epsilon_decay']
        self.epsilon_max = hyp_params['epsilon_initial']
        self.epsilon = self.epsilon_max
        self.epsilon_min = 0.01
        self.lr = hyp_params['lr']
        self.lr_decay_start = hyp_params['lr_decay_start']
        self.lr_decay_rate = hyp_params['lr_decay_rate']
        self.gamma = hyp_params['gamma']
        self.tau = hyp_params['tau']
        self.batch_size = hyp_params['batch_size']       
        self.memory_size = hyp_params['buffer_size']
        self.start_learning = hyp_params['start_learning']
        self.action_space = action_space
        self.observation_space = observation_space
        self.er = ExperienceReplay(self.memory_size)
        
        self.num_updates = 0 #number of training iterations
        
        logging.getLogger().setLevel(logging.INFO)
        
        self.Q_prime = self._build_network()
        self.Q_target = self._build_network()
        self.Q_target.set_weights(self.Q_prime.get_weights())
        self.q_optimizer = Adam(lr=self.lr)
        
    def _build_network(self):
        inputs = Input(shape=(self.observation_space, ), name='state')
        x = inputs
        for i in range(0,self.nn_params['num_layers']):
            x = Dense(self.nn_params['hidden_neurons'][i], 
                      activation=self.nn_params['activations'][i],
                      kernel_initializer=he_normal())(x)
            #x = BatchNormalization()(x)
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
    
    def update_learning_rate(self):
        if(self.num_updates >= self.lr_decay_start):
            self.lr = self.lr_decay_rate * self.lr
            self.q_optimizer = Adam(lr=self.lr)
            
    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
    
    def train_network(self):
        
        if self.er.buffer_size() < self.batch_size:
            return 0
        #get a batch from the memory buffer
        batch = self.er.get_batch(self.batch_size)
        states, actions, rewards, next_states, dones = \
            self.er.get_arrays_from_batch(batch, self.observation_space)
        
        #create indices needed for slicing
        batch_idxs = np.arange(len(batch))
        #cast as a int32 tensor and reshape
        batch_idxs = tf.cast(batch_idxs, tf.int32)
        batch_idxs = tf.reshape(batch_idxs,[len(batch_idxs),1])
        #cast actions as int32 tensor and reshape
        actions = tf.cast(actions, tf.int32)
        actions = tf.reshape(actions,[len(actions),1])
        
        #perform the update
        with tf.GradientTape() as tape:
            #get q_vals from primary network for s'
            q_vals_next_state = self.Q_prime(next_states)
            #compute optimal actions from Q(s',a)
            actions_optimal = tf.cast(tf.math.argmax(q_vals_next_state, axis=1), tf.int32)
            actions_optimal = tf.reshape(actions_optimal,[len(actions_optimal),1])
            
            #predict target Q values for s'
            q_next_state_target = self.Q_target(next_states)
            #create slices
            slices = tf.concat([batch_idxs, actions_optimal], axis=1)
            #compute the target
            y = rewards + self.gamma \
                * tf.gather_nd(q_next_state_target, slices) * (1 - dones)
            
            #compute Q values for s
            q_vals = self.Q_prime(states)
            #update the appropriate entries of q_vals with the updates
            q_update = q_vals
            slices = tf.concat([batch_idxs, actions], axis=1)
            q_update = tf.tensor_scatter_nd_update(q_vals, slices, y)
            #compute the loss (mse)
            q_loss = tf.math.reduce_mean(tf.math.square(q_vals - q_update))
        
        #compute the gradients
        q_grad = tape.gradient(q_loss, self.Q_prime.trainable_variables)
        #update the primary model
        self.q_optimizer.apply_gradients(
            zip(q_grad, self.Q_prime.trainable_variables)
        )
        
        #update target network (low pass filter)
        self.update_target_network()
        
        #update the learning rate of the optimizer
        self.update_learning_rate()
        
        #update the number of training sessions
        self.num_updates += 1
        return q_loss.numpy()

    def get_epsilon(self):
        return self.epsilon

    def train(self, env, episodes):
        #training loop
        ep_rewards = []
        ep_losses = []
        total_timesteps = 0
        for episode in range(episodes):
            rewards = []
            next_state = env.reset()
            done = False
            while not done:
                state = next_state
                action = self.get_action(next_state)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                rewards.append(reward)
                
                if(total_timesteps >= 500):
                    #train
                    ep_loss = self.train_network()
                    ep_losses.append(ep_loss)
                    #update epsilon (exploration)
                    self.update_epsilon()
                    
                total_timesteps += 1
                
            ep_rewards.append(sum(rewards))
            msg = "Episode: %03d, Reward: %03d, epsilon: %f, learning rate: %f"
            fmt = (episode, ep_rewards[-1], self.epsilon, self.lr)
            logging.info(msg % fmt)
                
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