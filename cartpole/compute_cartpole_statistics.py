# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:38:17 2020

@author: daniel arnold
"""

import csv
import cartpole as cart

num_episodes = 2000
epsilon_decay = 0.9995
verbose = False
make_plots = False
num_plays = 100

file_scores = "scores_" + str(epsilon_decay) + ".csv"
file_mean_scores = "mean_scores_" + str(epsilon_decay) + ".csv"
file_ticks_to_win = "ticks_to_win_" + str(epsilon_decay) + ".csv"

def save(file_name, record):
    csv_file = open(file_name,'a')
    csvWriter = csv.writer(csv_file, lineterminator='\n')

    csvWriter.writerow(record)


if __name__ == "__main__":
    
    for i in range(0,num_plays):
        scores,mean_scores,ticks_to_win = cart.cartpole(num_episodes, epsilon_decay, verbose, make_plots)
        print("iteration:", i, "ticks to win:", ticks_to_win)
        
        save(file_scores,scores)
        save(file_mean_scores,mean_scores)
        save(file_ticks_to_win,[ticks_to_win])
    