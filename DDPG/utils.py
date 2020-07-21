#Utility Functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#plot rewards, average rewards, and losses
def plot_results(rewards, avg_window, fig_name, 
                 default_val):
    #calculate rolling mean of rewards
    N = avg_window
    rewards_roll_avg = pd.Series(rewards).rolling(window=N).mean().iloc[N-1:].values
    rewards_roll_avg = np.concatenate((default_val * np.ones(N-1), rewards_roll_avg))
    
    fig, axs = plt.subplots(1,1)
    plt.style.use('seaborn')
    axs.plot(np.arange(0, len(rewards)), 
             rewards,
             label="raw score")
    axs.plot(np.arange(0, len(rewards)), 
             rewards_roll_avg,
             label="average_" + str(N))
    axs.set_xlabel('Episode')
    axs.set_ylabel('Total Reward')
    axs.legend()
    fig.suptitle(fig_name)
    plt.show()
