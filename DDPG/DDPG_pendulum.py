# -*- coding: utf-8 -*-
"""
@author: daniel arnold

inspired by: http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
and https://keras.io/examples/rl/ddpg_pendulum/
"""

import gym
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
#tensorflow inputs
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Concatenate
from keras.optimizers import Adam
import keras.backend as K
from keras.initializers import he_normal
from collections import deque
import random
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-n', '--num_updates', type=int, default=2000)
parser.add_argument('-r', '--render_test', action='store_true', default=False)
parser.add_argument('-p', '--plot_results', action='store_true', default=True)

#####################################
############ OU Noise  ##############
#####################################

class OUNoise:
    def __init__(self, mean, std, theta=0.15, dt=1e-2, x_init=None):
        self.theta = theta
        self.mean = mean
        self.std = std
        self.dt = dt
        self.x_init = x_init
        self.reset()
        
    def reset(self):
        if self.x_init is not None:
            self.x_prev = self.x_init
        else:
            self.x_prev = np.zeros_like(self.mean)
            
    def __call__(self):
        #xk = theta * (mean - xkm1) * dT + sigma*std*W
        x = (
            self.x_prev 
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        
        self.x_prev = x
        return x

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

class DQNAgent:
    def __init__(self, observation_space, action_space, action_low, 
                 action_high):
        
        self.lr_actor = 0.01
        self.lr_critic = 0.01
        self.gamma = 0.95
        self.tau = 0.08
        self.batch_size = 2000       
        self.memory_size = 20000
        self.action_space = action_space
        self.a_low = action_low
        self.a_high = action_high
        self.observation_space = observation_space
        self.er = ExperienceReplay(self.memory_size)
        self.noise = OUNoise(mean=np.zeros(1), std=0.2*np.ones(1))
        self.uniform_exploration_steps = 3000
        self.scale = 0.2
                
        #Q networks
        self.critic_prime = self._build_critic()
        self.critic_prime.compile(loss='mse', 
                             optimizer=Adam(lr=self.lr_critic))
        self.critic_target = self._build_critic()
        
        #Policy network
        self.actor_prime = self._build_actor()
        self.actor_prime.compile(loss=self.actor_loss, 
                             optimizer=Adam(lr=self.lr_actor))
        self.actor_target = self._build_actor()
        
    def _build_actor(self):
        #initialize weights between -3e-3 and 3e-3
        #init_weights = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
        inputs = Input(shape=(self.observation_space,))
        x = Dense(512, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(inputs)
        x = BatchNormalization()(x)
        #outputs = Dense(1, activation='tanh', 
        #               kernel_initializer=init_weights)(x)
        outputs = Dense(1, activation='tanh')(x)
        model = Model(inputs, outputs)
        return model
    
    def _build_critic(self):
        #Q network - has both the state and action as inputs
        #state inputs
        state_input = Input(shape=(self.observation_space,))
        x = Dense(16, activation='relu')(state_input)
        x = BatchNormalization()(x)
        x = Dense(32, activation='relu')(x)
        state_out = BatchNormalization()(x)
        
        #action inpust
        action_input = Input(shape=(self.action_space,))
        x = Dense(32, activation='relu')(action_input)
        action_out = BatchNormalization()(x)
        
        #both state and input are passed through seperate layers before concat
        combined_inputs = Concatenate()([state_out,action_out])
        
        x = Dense(512, activation='relu')(combined_inputs)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        outputs = Dense(1)(x)
        
        model = Model([state_input,action_input], outputs)
        return model
    
    def actor_loss(self, q_vals, actions):
        return -K.mean(q_vals)
                
    def update_target_networks(self):
        #critic
        weights = []
        for t,p in zip(self.critic_target.get_weights(), 
                       self.critic_prime.get_weights()):
            weights.append((1 - self.tau) * t + self.tau * p)
        self.critic_target.set_weights(weights)
        
        #actor
        weights = []
        for t,p in zip(self.actor_target.get_weights(), 
                       self.actor_prime.get_weights()):
            weights.append((1 - self.tau) * t + self.tau * p)
        self.actor_target.set_weights(weights)
    
    def remember(self, state, action, reward, next_state, done):
        self.er.store(state, action, reward, next_state, done)
        
    def get_action(self, state, step):
        #uniform exploration for first few steps
        if step <= self.uniform_exploration_steps:
            action = np.random.uniform(self.a_low, self.a_high)
        else:
            action = self.actor_prime.predict_on_batch(state[None,:])
            noise = np.random.normal(0, self.scale)
            action = action + noise
            action = np.clip(action, self.a_low, self.a_high)
        return action
    
    def get_action_greedy(self, state):
        action = self.actor_prime.predict_on_batch(state[None,:])
        #with no noise, the actions should always be admissible
        return action
    
    def train_networks(self):
        if self.er.buffer_size < self.batch_size:
            return 0, 0
        
        #get bacth from replay buffer
        batch = self.er.get_batch(self.batch_size)
        states, actions, rewards, next_states, dones = \
            self.er.get_arrays_from_batch(batch, self.observation_space)
        
        #predict next actions from actor target
        target_actions = self.actor_target.predict_on_batch(next_states)
        #predict qvals from next actions and next states
        target_qvals = self.critic_target.predict_on_batch([next_states, 
                                                            target_actions])
        #compute TD target
        y = rewards + self.gamma * np.squeeze(target_qvals) * (1 - dones)

        #train the critic network (MSE loss)
        critic_loss = self.critic_prime.train_on_batch([states, actions], y)
        #train the actor network
        #acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
        critic_qvals = self.critic_prime.predict_on_batch([states, actions])
        actor_loss = self.actor_prime.train_on_batch(states, critic_qvals)
        
        self.update_target_networks()

        return actor_loss, critic_loss

    def train_by_minibatch(self, env, batch_sz, updates):
        #training loop
        ep_rewards = []
        ep_actor_losses = []
        ep_critic_losses = []
        rewards = []
        next_state = env.reset()
        total_timesteps = 0
        for update in range(updates):
            for step in range(batch_sz):
                state = next_state
                action = self.get_action(next_state, 0)
                next_state, reward, done, _ = env.step([action])
                self.remember(state, action, reward, next_state, done)
                rewards.append(reward)

                total_timesteps += 1
                                
                if done:
                    ep_rewards.append(sum(rewards))
                    rewards = []
                    next_state = env.reset()
                    
                    msg = "Episode: %03d, Reward: %03d, update: %03d"
                    fmt = (len(ep_rewards)-1, ep_rewards[-1], update)
                    logging.info(msg % fmt)
            
            #sample the buffer and train the actor and critic networks
            actor_loss, critic_loss = self.train_networks()
            
            ep_actor_losses.append(actor_loss)
            ep_critic_losses.append(critic_loss)
            
            logging.debug("[%d/%d] actor loss: %s, critic loss: %s" 
                          % (update + 1, updates, actor_loss, critic_loss))
        return ep_rewards, ep_actor_losses, ep_critic_losses
 
    def test(self, env, render=False):
        state, done, episode_reward = env.reset(), False, 0
        while not done:
            action = self.get_action_greedy(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
        return episode_reward
    
#####################################
############## Main #################
#####################################

if __name__ == "__main__":
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)
    
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    action_low = env.action_space.low[0]
    action_high = env.action_space.high[0]
    
    #initalize the DQN Agent
    agent = DQNAgent(observation_space, action_space, action_low, action_high)
    
    print("Training model...")
    rewards_history, actor_losses, critic_losses = agent.train_by_minibatch(env, args.batch_size, args.num_updates)
    print("Training complete.  Testing...")
    print("Total Episode Reward: %d out of -2000" % agent.test(env, args.render_test))
    
    #Plot results
    #calculate rolling mean of rewards
    N = 100
    rewards_mean = pd.Series(rewards_history).rolling(window=N).mean().iloc[N-1:].values
    rewards_mean_test = rewards_mean
    def_val = -2000 #value at beginning of rolling mean (for indices < N)
    rewards_mean = np.concatenate((def_val * np.ones(N-1), rewards_mean))
    
    if args.plot_results:
        fig, axs = plt.subplots(1,2)
        plt.style.use('seaborn')
        axs[0].plot(np.arange(0, len(rewards_history)), 
                 rewards_history,
                 label="raw score")
        axs[0].plot(np.arange(0, len(rewards_history)), 
                 rewards_mean,
                 label="average_" + str(N))
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Total Reward')
        axs[0].legend()
        axs[1].plot(critic_losses)
        axs[1].set_xlabel('Update')
        axs[1].set_ylabel('Critic Loss')
        fig.suptitle(env_name)
        plt.show()