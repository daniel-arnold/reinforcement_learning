# reinforcement_learning

This repo contains solved reinforcement learning problems.  What's included:

 - cartpole: solving cartpole with DQN.  This repo uses keras and tensorflow to solve the cartpole problem from gym.  As DQN is an off-policy method, two environments are used.  1 environment is used for training, where the agent can take random actions and the second environment is for testing, in which the agent takes the greedy action according to the present action value function, Q(s,a)
