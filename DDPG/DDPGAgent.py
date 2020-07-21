# -*- coding: utf-8 -*-
"""
@author: daniel arnold

inspired by: http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
and https://keras.io/examples/rl/ddpg_pendulum/
"""

import logging
import numpy as np
#tensorflow inputs
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
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
        if(sample_num <  batch_size):
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

class DDPGAgent:
    def __init__(self, hyp_params, observation_space, action_space, action_low, 
                 action_high):
        
        self.lr_actor = hyp_params['lr_actor']
        self.lr_critic = hyp_params['lr_critic']
        self.gamma = hyp_params['gamma']
        self.tau = hyp_params['tau']
        self.batch_size = hyp_params['batch_size']       
        self.memory_size = hyp_params['memory_size']
        self.action_space = action_space
        self.a_low = action_low
        self.a_high = action_high
        self.observation_space = observation_space
        self.er = ExperienceReplay(self.memory_size)
        self.scale = hyp_params['scale']
                
        #Q networks
        self.critic_prime = self._build_critic()
        #self.critic_prime.compile(loss='mse', 
        #                     optimizer=Adam(lr=self.lr_critic))
        self.critic_optimizer = Adam(self.lr_critic)
        self.critic_target = self._build_critic()
        
        #Policy network
        self.actor_prime = self._build_actor()
        #self.actor_prime.compile(loss=self.actor_loss, 
        #                     optimizer=Adam(lr=self.lr_actor))
        self.actor_optimizer = Adam(self.lr_actor)
        self.actor_target = self._build_actor()
        
        self.actor_target.set_weights(self.actor_prime.get_weights())
        self.critic_target.set_weights(self.critic_prime.get_weights())
        
    def _build_actor(self):
        #initialize weights between -3e-3 and 3e-3
        init_weights = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        inputs = Input(shape=(self.observation_space,))
        x = Dense(512, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        outputs = Dense(1, activation='tanh', 
                       kernel_initializer=init_weights)(x)
        outputs = outputs * self.a_high
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

        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = tf.squeeze(self.actor_prime(tf_state))
        noise = np.random.normal(0, self.scale)
        action = action.numpy() + noise
        action = np.clip(action, self.a_low, self.a_high)
        return [action]
    
    def get_action_greedy(self, state):
        action = self.actor_prime.predict_on_batch(state[None,:])
        #with no noise, the actions should always be admissible
        return action
    
    def train_networks(self):
        
        num_samples = min(self.batch_size, self.er.buffer_size)
        
        #get bacth from replay buffer
        batch = self.er.get_batch(num_samples)
        states, actions, rewards, next_states, dones = \
            self.er.get_arrays_from_batch(batch, self.observation_space)
               
        #convert to tensors
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        rewards = tf.cast(rewards, dtype=tf.float32)
        rewards = tf.reshape(rewards, (len(rewards),1))
        next_states = tf.convert_to_tensor(next_states)
        ones_and_dones = tf.convert_to_tensor(1 - dones)
        ones_and_dones = tf.cast(ones_and_dones, dtype=tf.float32)
        ones_and_dones = tf.reshape(ones_and_dones, (len(ones_and_dones),1))
        
        #use gradient tape
        #critic (Q network)
        with tf.GradientTape() as tape:
            #compute actions for the target using next states
            target_actions = self.actor_target(next_states)
            #compute rewards
            y = rewards + \
                self.gamma * self.critic_target([next_states, target_actions]) * ones_and_dones
            #compute loss function
            critic_value = self.critic_prime([states, actions])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_prime.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_prime.trainable_variables)
        )

        #actor (policy network)
        with tf.GradientTape() as tape:
            actions = self.actor_prime(states)
            critic_value = self.critic_prime([states, actions])
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)
            
        actor_grad = tape.gradient(actor_loss, self.actor_prime.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_prime.trainable_variables)
        )
        
        self.update_target_networks()

        return actor_loss, critic_loss

    def train_by_episode(self, env, num_episodes):
        #training loop
        ep_rewards = []
        actor_losses = []
        critic_losses = []
        total_timesteps = 0
        for episode in range(0,num_episodes):
            rewards = []
            done = False
            next_state = env.reset()
            while not done:
                state = next_state
                action = self.get_action(next_state, total_timesteps)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                rewards.append(reward)

                total_timesteps += 1
                #train the network every timestep
                
                actor_loss, critic_loss = self.train_networks()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                logging.debug("Episode: %03d, actor loss: %s, critic loss: %s" 
                          % (episode, actor_loss, critic_loss))
                                
            ep_rewards.append(sum(rewards))
            msg = "Episode: %03d, Reward: %03d"
            fmt = (episode, sum(rewards))
            logging.info(msg % fmt)
        return ep_rewards, actor_losses, critic_losses
 
    def test(self, env, render=False):
        state, done, episode_reward = env.reset(), False, 0
        while not done:
            action = self.get_action_greedy(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
        return episode_reward