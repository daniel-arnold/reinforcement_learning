
import numpy as np
import gym
from gym import wrappers
import pybullet_envs
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_episodes', type=int, default=1000)

#####################################
############# Classes################
#####################################

#class for normalizing the observations
class Normalizer():
    #Welford's online algorithm
    #implementation from: iamsuvhro
    def __init__(self, n_inputs):
        self.mean = np.zeros(n_inputs)
        self.n = np.zeros(n_inputs)
        self.sos_diff = np.zeros(n_inputs)
        self.var = np.zeros(n_inputs)

    def update_statistics(self, x):
        self.n += 1
        #update mean 
        last_mean = self.mean.copy()
        self.mean += (x - self.mean)/self.n
        #update sum of squares differences
        self.sos_diff += (x-last_mean)*(x-self.mean)
        self.var = (self.sos_diff/self.n).clip(min=1e-2)

    def normalize(self, u):
        self.update_statistics(u)
        u_no_mean = u - self.mean
        u_std = np.sqrt(self.var)
        return u_no_mean/u_std

#class for ARS agent
class ARSAgent():
    def __init__(self, num_episodes):
        self.alpha = 0.02 #learning rate
        self.mu = 0.03 #exploration noise
        self.num_directions = 16 #number of random directions to consider
        self.num_best_directions = 16 #number of best directions to consider
        assert self.num_best_directions <= self.num_directions
        self.max_iterations = num_episodes #number of iterations
        self.max_episode_steps = 2000 #max steps in episode
        self.env_name = 'HalfCheetahBulletEnv-v0'
        self.env = gym.make(self.env_name)
        self.n_inputs = self.env.observation_space.shape[0]
        self.n_outputs = self.env.action_space.shape[0]
        #self.seed = 1
        #np.random.seed(self.seed)
        self.theta = np.zeros((self.n_inputs,self.n_outputs))
        self.normalizer = Normalizer(self.n_inputs)

    def get_action(self, state, theta):
        _u = np.dot(theta.T, state)
        return _u

    def rollout(self, theta):
        state = self.env.reset()
        #rollout for episode:
        done = False
        sum_rewards = 0
        k = 0
        while not done and k<self.max_episode_steps:
            #normalize state
            state = self.normalizer.normalize(state)
            #get next action
            u = self.get_action(state, theta)
            state, reward, done, _ = self.env.step(u)
            reward = max(min(reward, 1), -1)
            sum_rewards += reward
            k+=1
        return sum_rewards

    def random_directions(self):
        return [np.random.randn(*self.theta.shape) for _ in range(self.num_directions)]

    def random_search(self):
        #run 1 iteration of augmented random search
        d = self.random_directions()
        r_pos = []
        r_neg = []
        for i in range(0,self.num_directions):
            #generate random direction
            _d = d[i]
            #rollout in _d and -_d
            theta_d_pos = self.theta + self.mu * _d
            theta_d_neg = self.theta - self.mu * _d
            #compute positive and negative rewards
            r_pos.append(self.rollout(theta_d_pos))
            r_neg.append(self.rollout(theta_d_neg))

        #compute std for rewards
        r_std = np.asarray(r_pos + r_neg).std()

        #find indices of best b rewards
        best_scores = [max(_r_pos, _r_neg) for k,(_r_pos,_r_neg) in enumerate(zip(r_pos, r_neg))]
        idxs = np.asarray(best_scores).argsort()[-self.num_best_directions:]
        #GD
        _theta = np.zeros(self.theta.shape)
        for idx in list(idxs):
            _theta += self.alpha/self.num_best_directions * (r_pos[idx] - r_neg[idx])/r_std * d[idx]
        #update theta
        self.theta += _theta
        #rollout with the new policy for evaluation
        r_eval = self.rollout(self.theta)
        return self.theta, r_eval

    def train(self):
        k=0
        thetas = []
        rewards = []
        while k<self.max_iterations:
            #run step of ARS
            _theta, _r = self.random_search()
            thetas.append(_theta)
            rewards.append(_r)
            print("Iteration: ", k, " ---------- reward: ", _r)
            k+=1
        return thetas,rewards
    
#####################################
############## Main #################
#####################################

if __name__ == "__main__":
    args = parser.parse_args()

    n = args.num_episodes
    #create ARS agent
    ars = ARSAgent(n)
    #train the agent
    thetas,rewards = ars.train()

    #store results in dataframe and save to csv
    rewards_dict = {'rewards':rewards}
    rewards_df = pd.DataFrame(rewards_dict)
    rewards_df.to_csv('ARS_cheetah_rewards_' + str(n) +'.csv')

    #save best weights to csv
    thetas_dict = {'theta':thetas[-10:-1]}
    thetas_df = pd.DataFrame(thetas_dict)
    thetas_df.to_csv('ARS_cheetah_thetas_' + str(n) +'.csv')