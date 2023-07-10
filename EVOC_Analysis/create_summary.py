# Creates VOC summary data
#
# Groups by voc bin and calculates average response time

# load packages
import sys
import os
import numpy as np
import pandas as pd
import scipy.stats as stats 
import matplotlib.pyplot as plt
import seaborn as sns
import chess
import stockfish
from stockfish import Stockfish
import chess.engine
import chess.pgn
from datetime import datetime
from multiprocessing import Pool
import chess.pgn
from multiprocessing import util

SOURCE_FOLDER = "./data/clean/pkl_v2/"
SAVE_FILE = './data/clean/all_summary_20221003.pkl'

# time controls to process
time_ctrls = ['60+0','120+1','180+0','180+2','300+0','300+3',
                  '600+0','600+5', '900+10', '1800+0', '1800+20']

# define bins
VOC_BINS = np.linspace(-0.02,0.42,23)
VOC_BINS_c = (VOC_BINS[1:] + VOC_BINS[:-1])/2
VOC_BINS_c[0] = 0

df_list = []
for i in range(len(time_ctrls)):
    
    # load files and clip by ply
    load_file = SOURCE_FOLDER + time_ctrls[i].replace("+", "_" ) + "_full.pkl"    
    df_f = pd.read_pickle(load_file) 
    df = df_f[(15 <= df_f.move_ply) & (df_f.move_ply <= 75)]
       
    # for each type of voc calculation bin and calculate bin averages
    for voc_type in ['evan', 'voc']:
        bins = VOC_BINS
        g = pd.cut(df[voc_type], bins)
        means = df.groupby(g)['rt'].mean().reset_index().rename(columns = {'rt':'mean'})
        stds = df.groupby(g)['rt'].std().reset_index().rename(columns = {'rt':'std'})
        counts = df.groupby(g)['rt'].size().reset_index().rename(columns = {'rt':'count'})
        df_p = pd.concat([counts, means['mean'], stds['std']], axis=1)
        df_p['ctrl'] = time_ctrls[i]
        df_p['bin_c'] = VOC_BINS_c
        df_p['type'] = voc_type

        df_list.append(df_p) 
            
    df_sum = pd.concat(df_list).reset_index(drop=True)
    
    df_sum.to_pickle(SAVE_FILE)
    
    
    
# summarize cases where realized voc is 0    
df = pd.read_pickle("./data/clean/pkl_v2/600_0_full.pkl")
VOC_BINS2 = np.linspace(0,0.44,23)
VOC_BINS2_c = np.linspace(0.01,0.43,22)
voc_type = 'voc'
bins = VOC_BINS2
#voc_type = 'voc'
g = pd.cut(df[voc_type], bins)
means = df.groupby(g)['rt'].mean().reset_index().rename(columns = {'rt':'mean'})
stds = df.groupby(g)['rt'].std().reset_index().rename(columns = {'rt':'std'})
counts = df.groupby(g)['rt'].size().reset_index().rename(columns = {'rt':'count'})
df_p = pd.concat([counts, means['mean'], stds['std']], axis=1)
df_p['ctrl'] = '600+0'
df_p['bin_c'] = VOC_BINS2_c
df_p['type'] = 'voc_evan0'

df_list.append(df_p) 
            
df_sum = pd.concat(df_list).reset_index(drop=True)

df_sum.to_pickle('./data/clean/all_summary_20221003.pkl')
