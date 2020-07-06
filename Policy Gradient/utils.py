#Utility Functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#plot rewards, average rewards, and losses
def plot_results(rewards, losses, avg_window, fig_name, default_val):
    #calculate rolling mean of rewards
    N = avg_window
    rewards_roll_avg = pd.Series(rewards).rolling(window=N).mean().iloc[N-1:].values
    rewards_roll_avg = np.concatenate((default_val * np.ones(N-1), rewards_roll_avg))
    
    fig, axs = plt.subplots(1,2)
    plt.style.use('seaborn')
    axs[0].plot(np.arange(0, len(rewards)), rewards, label="score")
    axs[0].plot(np.arange(0, len(rewards)), rewards_roll_avg, label="average_" + str(N))
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total Reward')
    axs[0].legend()
    axs[1].plot(losses)
    axs[1].set_xlabel('Update')
    axs[1].set_ylabel('Loss')
    fig.suptitle(fig_name)
    plt.show()
