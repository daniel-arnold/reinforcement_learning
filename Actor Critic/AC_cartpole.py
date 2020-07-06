# -*- coding: utf-8 -*-
"""
Solves Cartpole-v1 with AC
"""

import gym
import argparse
from utils import plot_results
from ACAgent import CategoricalACAgent

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=128)
parser.add_argument('-n', '--num_updates', type=int, default=1000)
parser.add_argument('-lra', '--learning_rate_actor', type=float, default=7e-4)
parser.add_argument('-lrc', '--learning_rate_critic', type=float, default=7e-4)
parser.add_argument('-r', '--render_test', action='store_true', default=False)
parser.add_argument('-p', '--plot_results', action='store_true', default=True)

if __name__ == "__main__":
    args = parser.parse_args()
    
    #Actor MLP NN
    nn_actor = {}
    num_layers = 1
    hidden_layer_neurons = ['128']
    activations = ['relu']
    nn_actor['num_layers'] = num_layers
    nn_actor['hidden_neurons'] = hidden_layer_neurons
    nn_actor['activations'] = activations
    
    #Critic MLP NN
    nn_critic = {}
    num_layers = 1
    hidden_layer_neurons = ['128']
    activations = ['relu']
    nn_critic['num_layers'] = num_layers
    nn_critic['hidden_neurons'] = hidden_layer_neurons
    nn_critic['activations'] = activations
    
    
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    agent = CategoricalACAgent(nn_actor, nn_critic, 
                               env.observation_space.shape[0], 
                               env.action_space.n, 
                               args.learning_rate_actor, 
                               args.learning_rate_critic)
    
    print("Training model...")
    rewards, critic_losses, actor_losses = agent.train(env, args.batch_size, 
                                                       args.num_updates)
    print("Training complete.  Testing...")
    print("Total Episode Reward: %d out of 500" % agent.test(env, args.render_test))
    
    #Plot results
    plot_results(rewards, actor_losses, critic_losses, 100, env_name, 0)