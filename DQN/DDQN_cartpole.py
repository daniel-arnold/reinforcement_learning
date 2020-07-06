# -*- coding: utf-8 -*-
"""
@author: daniel arnold

inspired by: http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
and https://rubikscode.net/2020/01/27/double-dqn-with-tensorflow-2-and-tf-agents-2/
"""

import gym
import argparse
from utils import plot_results
from DDQNAgent import DDQNAgent

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-n', '--num_updates', type=int, default=4000)
parser.add_argument('-r', '--render_test', action='store_true', default=False)
parser.add_argument('-p', '--plot_results', action='store_true', default=True)

if __name__ == "__main__":
    args = parser.parse_args()
    
    #Actor MLP NN
    nn_params = {}
    num_layers = 1
    hidden_layer_neurons = ['128']
    activations = ['relu']
    nn_params['num_layers'] = num_layers
    nn_params['hidden_neurons'] = hidden_layer_neurons
    nn_params['activations'] = activations
    
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    
    #initalize the DQN Agent
    hyp_params = {}
    #clipnorm value in optimizer, negative values means clipnorm will not be used
    hyp_params['clipnorm_val'] = 1.0
    hyp_params['tau'] = 0.08
    hyp_params['gamma'] = 0.99
    hyp_params['epsilon_decay'] = 0.9998
    hyp_params['lr'] = 0.01
    hyp_params['batch_size'] = 2000
    hyp_params['memory_size'] = 20000
    
    agent = DDQNAgent(nn_params, hyp_params, observation_space, action_space)
    
    print("Training model...")
    rewards, losses = agent.train(env, args.batch_size, 
                                          args.num_updates)
    print("Training complete.  Testing...")
    print("Total Episode Reward: %d out of 500" % agent.test(env, args.render_test))
    
    #Plot results
    plot_results(rewards, losses, 100, env_name, 0)
