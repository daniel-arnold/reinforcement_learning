# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 21:34:14 2020

@author: daniel arnold

inspired by: http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
"""

import gym
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-n', '--num_updates', type=int, default=250)
parser.add_argument('-lr', '--learning_rate', type=float, default=7e-3)
parser.add_argument('-r', '--render_test', action='store_true', default=False)
parser.add_argument('-p', '--plot_results', action='store_true', default=False)

#####################################
############## Model ################
#####################################

class ProbDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        #draw an action from a categorical (discrete) distribution:
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Model(tf.keras.Model):
    def __init__(self, action_space):
        super().__init__('mlp_policy')
        self.hidden1 = kl.Dense(128, activation='relu')
        #output layers
        self.logits = kl.Dense(action_space, name='policy_logits')
        self.dist = ProbDistribution()
        
    def call(self, inputs, **kwargs):
        #inputs are numpy array, that we convert to a tensor
        x = tf.convert_to_tensor(inputs)
        hidden_logits = self.hidden1(x)
        return self.logits(hidden_logits)
    
    def get_action(self, obs):
        #compute logits and value
        logits = self.predict_on_batch(obs)
        #compute actions
        action = self.dist.predict_on_batch(logits)
        return np.squeeze(action, axis=-1)

#####################################
############## Agent ################
#####################################  

class A2CAgent:
    def __init__(self, model, lr=7e-3, gamma=0.99, value_c=0.5, entropy_c=1e-4):
        self.lr = lr
        self.gamma = gamma
        self.value_c = value_c
        self.entropy_c = entropy_c
        
        self.model = model
        self.model.compile(
            optimizer=ko.Adam(lr=self.lr),
            loss=self._logits_loss)


    def train(self, env, batch_sz=64, updates=300):
        #need to account for the fact that the batch size may 
        #cross episodes
        
        #storage helpers for single batch of data
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones = np.empty((2, batch_sz))
        observations = np.empty((batch_sz,)  + env.observation_space.shape)
        #training loop
        ep_rewards = [0.0]
        next_obs = env.reset()
        for update in range(updates):
            for step in range(batch_sz):
                observations[step] = next_obs.copy()
                #get the action and value for this state
                actions[step] = self.model.get_action(next_obs[None, :])
                #advance the environment
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])
                
                ep_rewards[-1] += rewards[step]
                if dones[step]:
                    ep_rewards.append(0.0)
                    next_obs = env.reset()
                    logging.info("Episode: %03d, Reward: %03d" % (len(ep_rewards)-1, ep_rewards[-2]))
            
            #if episode isn't done, will need V(s_tp1) for advantage
            returns, advs = self._returns_advantages(rewards, dones)
            #trick to input actions and advantages into same API
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            #train with the batch
            #losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
            losses = self.model.train_on_batch(observations, acts_and_advs)
            logging.debug("[%d/%d] Losses: %s" % (update + 1, updates, losses))
            
        return ep_rewards
 
    def test(self, env, render=False):
        state, done, episode_reward = env.reset(), False, 0
        while not done:
            action = self.model.get_action(state[None, :])
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
        return episode_reward
    
    def _returns_advantages(self, rewards, dones):
        returns = rewards.copy()
        #super clever way to do this...
        for t in reversed(range(rewards.shape[0] - 1)):
            returns[t] = rewards[t] + self.gamma * returns[t+1] * (1 - dones[t])
        #compute advantages = returns - baseline
        baseline = np.mean(returns) * np.ones_like(returns)
        disc_returns = (returns - baseline)
        advantages = disc_returns / np.std(disc_returns)
        return disc_returns, advantages
    
    def _logits_loss(self, actions_and_advantages, logits):
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
        #sparse categorical ce loss call has args: y_true, y_pred, sample_weights
        #from_logits ensures transformation is normalized into probabilities
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        #policy loss is policy gradient weighted by advantages
        #only calculate loss on actions actually taken
        #transform numpy actions into tensor
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        #calculate entropy loss: H = sum(p(x)*log(q(x)))
        probs = tf.nn.softmax(logits)
        entropy_loss = kls.categorical_crossentropy(probs, probs)
        #goal is to minimize policy loss and maximze entropy loss
        return policy_loss - self.entropy_c * entropy_loss
    
#####################################
############## Main #################
#####################################

if __name__ == "__main__":
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)
    
    env = gym.make('CartPole-v0')
    model = Model(action_space=env.action_space.n)
    agent = A2CAgent(model, args.learning_rate)
    
    print("Training model...")
    rewards_history = agent.train(env, args.batch_size, args.num_updates)
    print("Training complete.  Testing...")
    print("Total Episode Reward: %d out of 200" % agent.test(env, args.render_test))
    
    if args.plot_results:
        plt.style.use('seaborn')
        plt.plot(np.arange(0, len(rewards_history), 10), rewards_history[::10])
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.show()