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
parser.add_argument('-b', '--batch_size', type=int, default=128)
parser.add_argument('-e', '--max_epochs', type=int, default=300)
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
        # we'll use one network with multiple outputs for actor and critic
        #hidden layers for AC
        self.hidden1 = kl.Dense(128, activation='relu')
        self.hidden2 = kl.Dense(128, activation='relu')
        #output layers
        self.value = kl.Dense(1, name="value")
        self.logits = kl.Dense(action_space, name='policy_logits')
        self.dist = ProbDistribution()
        
    def call(self, inputs, **kwargs):
        #inputs are numpy array, that we convert to a tensor
        x = tf.convert_to_tensor(inputs)
        hidden_logits = self.hidden1(x)
        hidden_value = self.hidden2(x)
        return self.logits(hidden_logits), self.value(hidden_value)
    
    def get_action_value(self, obs):
        #compute logits and value
        logits, value = self.predict_on_batch(obs)
        #compute actions
        action = self.dist.predict_on_batch(logits)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

#####################################
############## Agent ################
#####################################  

class A2CAgent:
    def __init__(self, model, lr=7e-3, gamma=0.99, value_c=0.5, entropy_c=1e-1):
        self.lr = lr
        self.gamma = gamma
        self.value_c = value_c
        self.entropy_c = entropy_c
        
        self.model = model
        self.model.compile(
            optimizer=ko.Adam(lr=self.lr),
            loss=[self._logits_loss, self._value_loss])


    def train(self, env, batch_sz, max_epochs):
        #need to account for the fact that the batch size may 
        #cross episodes
        
        #storage helpers for single batch of data
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,)  + env.observation_space.shape)
        #training loop
        ep_rewards = [0.0]
        next_obs = env.reset()
        epoch = 0
        #for update in range(updates):
        while(epoch < max_epochs):
            for step in range(batch_sz):
                observations[step] = next_obs.copy()
                #get the action and value for this state
                actions[step], values[step] = self.model.get_action_value(next_obs[None, :])
                #advance the environment
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])
                
                ep_rewards[-1] += rewards[step]
                if dones[step]:
                    ep_rewards.append(0.0)
                    next_obs = env.reset()
                    logging.info("Episode: %03d, Reward: %03d" % (len(ep_rewards)-1, ep_rewards[-2]))
                    epoch += 1
            
            #if episode isn't done, will need V(s_tp1) for advantage
            _, next_value = self.model.get_action_value(next_obs[None, :])
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            #trick to input actions and advantages into same API
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            #acts_and_advs = np.concatenate([actions[None, :], advs[None, :]], axis=-1)
            #train with the batch
            losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
            logging.debug("[%d/%d] Losses: %s" % (epoch + 1, max_epochs, losses))
            
        return ep_rewards
 
    def test(self, env, render=False):
        state, done, episode_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.get_action_value(state[None, :])
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
        #minimize mse between estimates and returns
        return self.value_c * kls.mean_squared_error(returns, values)
    
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
    
    env = gym.make('Acrobot-v1')
    model = Model(action_space=env.action_space.n)
    agent = A2CAgent(model, args.learning_rate)
    
    print("Training model...")
    rewards_history = agent.train(env, args.batch_size, args.max_epochs)
    print("Training complete.  Testing...")
    print("Total Episode Reward: %d" % agent.test(env, args.render_test))
    
    if args.plot_results:
        plt.style.use('seaborn')
        plt.plot(np.arange(0, len(rewards_history)), rewards_history, label='rewards')
        plt.plot(np.arange(0, len(rewards_history), 10), rewards_history[::10], label='average_rewards(10)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Rewards in Training')
        plt.legend()
        plt.show()