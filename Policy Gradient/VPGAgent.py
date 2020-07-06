import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import tensorflow.keras.optimizers as ko
import tensorflow.keras.backend as K

#####################################
############## Agent ################
#####################################  

class CategoricalVPGAgent:
    def __init__(self, nn_params, observation_space, action_space, 
                 lr, gamma=0.99, value_c=0.5,
                 entropy_c=1e-4):

        self.lr = lr
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.value_c = value_c
        self.entropy_c = entropy_c
        
        #set logging level
        logging.getLogger().setLevel(logging.INFO)
        
        #policy model
        inputs = Input(shape=(self.observation_space, ), name='state')
        x = inputs
        
        for i in range(0,nn_params['num_layers']):
            x = Dense(nn_params['hidden_neurons'][i], 
                      activation=nn_params['activations'][i])(x)
        
        logits = Dense(self.action_space, activation='linear')(x)
        self.logits = Model(inputs, logits, name='policy')
        self.logits.compile(loss=self._logits_loss, 
                            optimizer=ko.Adam(lr=self.lr))
        
    def get_action(self, obs):
        #compute logits and value
        logits = self.logits.predict_on_batch(obs)
        #compute actions
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
        return np.squeeze(action, axis=-1)

    def train_by_step(self, env, batch_sz, updates):
        #need to account for the fact that the batch size may 
        #cross episodes
        
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones = np.empty((2, batch_sz))
        observations = np.empty((batch_sz,)  + env.observation_space.shape)
        #training loop
        ep_rewards = [0.0]
        ep_losses = []
        next_obs = env.reset()
        for update in range(updates):
            for step in range(batch_sz):
                observations[step] = next_obs.copy()
                #get the action and value for this state
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
            returns, advs = self._returns_advantages(rewards, dones)
            #trick to input actions and advantages into same API
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            #train with the batch
            losses = self.logits.train_on_batch(observations, acts_and_advs)
            ep_losses.append(losses)
            logging.debug("[%d/%d] Losses: %s" % (update + 1, updates, losses))
            
        return ep_rewards, ep_losses
    
    def train_by_episode(self, env, episodes):

        #training loop
        ep_rewards = []
        ep_losses = []
        for episode in range(episodes):
            actions, rewards, states = [], [], []
            next_state = env.reset()
            done = False
            while not done:
                state = next_state
                action = self.get_action(state[None, :])
                next_state, reward, done, _ = env.step(action)
                actions.append(action)
                rewards.append(reward)
                states.append(state)
            
            ep_rewards.append(sum(rewards))
            logging.info("Episode: %03d, Reward: %03d" % (episode+1, ep_rewards[-1]))
            
            actions = np.asarray(actions)
            rewards = np.asarray(rewards)
            states = np.asarray(states)

            returns, advs = self._returns_advantages(rewards, np.zeros_like(rewards))
            #trick to input actions and advantages into same API
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            #train with the batch
            loss = self.logits.train_on_batch(states, acts_and_advs)
            logging.debug("[%d/%d] loss: %s " 
                          % (episode + 1, episodes, loss))
            ep_losses.append(loss)
        return ep_rewards, ep_losses
 
    def test(self, env, render=False):
        state, done, episode_reward = env.reset(), False, 0
        while not done:
            action = self.get_action(state[None, :])
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
        return episode_reward
    
    def _returns_advantages(self, rewards, dones):
        returns = rewards.copy()
        for t in reversed(range(rewards.shape[0] - 1)):
            returns[t] = rewards[t] + self.gamma * returns[t+1] * (1 - dones[t])
        #compute advantages = returns - baseline
        baseline = np.mean(returns) * np.ones_like(returns)
        disc_returns = (returns - baseline)
        advantages = disc_returns / np.std(disc_returns)
        return disc_returns, advantages
    
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