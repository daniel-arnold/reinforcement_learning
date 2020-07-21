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
parser.add_argument('-n', '--num_episodes', type=int, default=200)
parser.add_argument('-r', '--render_test', action='store_true', default=False)
parser.add_argument('-p', '--plot_results', action='store_true', default=True)

if __name__ == "__main__":
    args = parser.parse_args()
    
    #MLP NN parameters
    nn_params = {}
    num_layers = 2
    hidden_layer_neurons = ['64','64']
    activations = ['relu','relu']
    nn_params['num_layers'] = num_layers
    nn_params['hidden_neurons'] = hidden_layer_neurons
    nn_params['activations'] = activations
    
    env_name = 'Acrobot-v1'
    env = gym.make(env_name)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    
    #initalize the DQN Agent
    hyp_params = {}
    #clipnorm value in optimizer, negative values means clipnorm will not be used
    hyp_params['tau'] = 0.1
    hyp_params['gamma'] = 0.99
    hyp_params['epsilon_decay'] = 0.99
    hyp_params['epsilon_initial'] = 0.5
    hyp_params['epsilon_min'] = 0.01
    hyp_params['lr'] = 0.0075
    hyp_params['lr_decay_start'] = 2000
    hyp_params['lr_decay_rate'] = 0.9995
    hyp_params['batch_size'] = 400
    hyp_params['buffer_size'] = 20000
    hyp_params['start_learning'] = 500
    
    agent = DDQNAgent(nn_params, hyp_params, observation_space, action_space)
    
    print("Training model...")
    rewards, losses = agent.train(env, args.num_episodes)
    #rewards, losses = agent.train(env, args.max_episodes)
    print("Training complete.  Testing...")
    print("Total Episode Reward: %d out of 0" % agent.test(env, args.render_test))
    
    #Plot results
    plot_results(rewards, losses, 100, env_name, -500)
