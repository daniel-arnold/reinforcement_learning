# -*- coding: utf-8 -*-
"""
@author: daniel arnold

inspired by: http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
"""

import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import tensorflow.keras.optimizers as ko
import tensorflow.keras.backend as K 

class CategoricalACAgent:
    def __init__(self, nn_params_actor, nn_params_critic, 
                 observation_space, action_space, 
                 lr_actor, lr_critic, gamma=0.99, value_c=0.5, 
                 entropy_c=1e-4):
        
        self.action_space = action_space
        self.observation_space = observation_space
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.value_c = value_c
        self.entropy_c = entropy_c
        
        logging.getLogger().setLevel(logging.INFO)
        
        inputs = Input(shape=(self.observation_space, ), name='state')
        #value model
        x = inputs
        for i in range(0,nn_params_critic['num_layers']):
            x = Dense(nn_params_critic['hidden_neurons'][i], 
                      activation=nn_params_critic['activations'][i])(x)
        value = Dense(1, activation='linear')(x)
        self.value = Model(inputs, value, name='value')
        self.value.compile(loss=self._value_loss, 
                            optimizer=ko.Adam(lr=self.lr_critic))
        #policy model
        x = inputs
        for i in range(0,nn_params_actor['num_layers']):
            x = Dense(nn_params_actor['hidden_neurons'][i], 
                      activation=nn_params_actor['activations'][i])(x)
        logits = Dense(self.action_space, activation='linear')(x)
        self.logits = Model(inputs, logits, name='policy')
        self.logits.compile(loss=self._logits_loss, 
                            optimizer=ko.Adam(lr=self.lr_actor))
        
    def get_value(self, obs):
        #compute logits and value
        value = self.value.predict_on_batch(obs)
        return np.squeeze(value, axis=-1)
    
    def get_action(self, obs):
        #compute logits and value
        logits = self.logits.predict_on_batch(obs)
        #compute actions
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
        return np.squeeze(action, axis=-1)

    def train(self, env, batch_sz, updates):
        #need to account for the fact that the batch size may 
        #cross episodes
        
        #storage helpers for single batch of data
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,)  + env.observation_space.shape)
        #training loop
        ep_rewards = [0.0]
        ep_value_losses = []
        ep_logit_losses = []
        next_obs = env.reset()
        for update in range(updates):
            for step in range(batch_sz):
                observations[step] = next_obs.copy()
                #get the action and value for this state
                values[step] = self.get_value(next_obs[None, :])
                actions[step] = self.get_action(next_obs[None, :])
                #advance the environment
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])
                
                ep_rewards[-1] += rewards[step]
                if dones[step]:
                    ep_rewards.append(0.0)
                    next_obs = env.reset()
                    msg = "Episode: %03d, Reward: %03d, update: %03d"
                    fmt = (len(ep_rewards)-1, ep_rewards[-2], update)
                    logging.info(msg % fmt)
            
            #if episode isn't done, will need V(s_tp1) for advantage
            next_value = self.get_value(next_obs[None, :])
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            #trick to input actions and advantages into same API
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            #train with the batch
            value_loss = self.value.train_on_batch(observations, returns)
            logits_loss = self.logits.train_on_batch(observations, acts_and_advs)
            logging.debug("[%d/%d] value loss: %s logits loss: %s " 
                          % (update + 1, updates, value_loss, logits_loss))
            ep_value_losses.append(value_loss)
            ep_logit_losses.append(logits_loss)
        return ep_rewards, ep_value_losses, ep_logit_losses
 
    def test(self, env, render=False):
        state, done, episode_reward = env.reset(), False, 0
        while not done:
            action = self.get_action(state[None, :])
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
        return episode_reward
    
    def _returns_advantages(self, rewards, dones, values, next_value):
    
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        #Rt = rt + gamma * V(s_t+1)
        #super clever way to do this...
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t+1] * (1 - dones[t])
        returns = returns[:-1]
        #compute advantages = returns - V
        advantages = returns - values
        return returns, advantages
    
    def _value_loss(self, returns, values):
        #MSE loss
        value_loss = K.mean(K.flatten(K.square(returns - values)))
        return self.value_c * value_loss
    
    def _logits_loss(self, actions_and_advantages, logits):
        #cross entropy loss
        #split actions and advantages
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
        #actions should be integers (sparse encoding)
        actions = tf.cast(actions, tf.int32)
        #flatten actions and advantages
        actions = K.flatten(actions)
        advantages = K.flatten(advantages)
        #convert sparse encoding to one hot
        actions_ones = K.one_hot(actions,self.action_space)
        #calculate probs from logits
        probs = tf.nn.softmax(logits)
        #calculate log probs from logits
        log_probs = tf.math.log(probs)
        #calculate policy gradients
        p_grads = K.sum((actions_ones * log_probs), axis=1)
        #calculate policy loss (entry-wise multiplication with advantages)        
        p_loss = -K.mean(p_grads * advantages)
        #entropy
        entropy_loss = -K.mean(K.sum(probs * log_probs, axis=1))
        return p_loss + self.entropy_c * entropy_loss