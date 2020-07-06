# -*- coding: utf-8 -*-
"""
@author: daniel arnold

inspired by: http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
"""

import gym
import argparse
from VPGAgent import CategoricalVPGAgent
from utils import plot_results

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=128)
parser.add_argument('-n', '--num_updates', type=int, default=1000)
parser.add_argument('-lr', '--learning_rate', type=float, default=7e-4)
parser.add_argument('-r', '--render_test', action='store_true', default=False)
parser.add_argument('-p', '--plot_results', action='store_true', default=True)
    
#####################################
############## Main #################
#####################################

if __name__ == "__main__":
    args = parser.parse_args()
    #parametres for MLP NN
    nn_params = {}
    num_layers = 1
    hidden_layer_neurons = ['128']
    activations = ['relu']
    
    nn_params['num_layers'] = num_layers
    nn_params['hidden_neurons'] = hidden_layer_neurons
    nn_params['activations'] = activations
    
    #setup environment
    env_name = 'Acrobot-v1'
    env = gym.make(env_name)
    
    #instantiate agent
    agent = CategoricalVPGAgent(nn_params, env.observation_space.shape[0], 
                                env.action_space.n, args.learning_rate, entropy_c=1e-2)
    #Train and test the model
    print("Training model...")
    rewards_history, losses = agent.train_by_step(env, args.batch_size, args.num_updates)
    print("Training complete.  Testing...")
    print("Total Episode Reward: %d out of -500" % agent.test(env, args.render_test))
    
    #plot
    plot_results(rewards_history, losses, 100, env_name, -500)