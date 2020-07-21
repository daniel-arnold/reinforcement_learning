# -*- coding: utf-8 -*-
"""
@author: daniel arnold

inspired by: http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
and https://keras.io/examples/rl/ddpg_pendulum/
"""

import gym
import logging
import argparse
from utils import plot_results
from DDPGAgent import DDPGAgent

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_episodes', type=int, default=150)
parser.add_argument('-r', '--render_test', action='store_true', default=False)
parser.add_argument('-p', '--plot_results', action='store_true', default=True)
    
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
    
    #hyper parameters
    hyp_params = {}
    hyp_params['tau'] = 0.005
    hyp_params['gamma'] = 0.99
    hyp_params['lr_actor'] = 0.001
    hyp_params['lr_critic'] = 0.002
    hyp_params['batch_size'] = 64
    hyp_params['memory_size'] = 50000
    hyp_params['scale'] = 0.2
    
    agent = DDPGAgent(hyp_params, observation_space, 
                      action_space, action_low, action_high)
    
    print("Training model...")
    rewards_history, actor_losses, critic_losses = agent.train_by_episode(env, args.num_episodes)
    print("Training complete")
    
    if args.plot_results:
        plot_results(rewards_history, 25, env_name, -2000)