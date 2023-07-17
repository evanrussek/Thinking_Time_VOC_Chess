# Creates a striped down pickle file with basic game data and VOC
#
# Mapping from centipawn scores (cp) to win percentage is done here
# to avoid having to reprocess individual games.


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

from convert_stockfish_scores import *

RAW_FOLDERS = ['data/raw/csv_jan', 'data/raw/csv_feb', 'data/raw/csv_mar', 'data/raw/csv_apr']
SAVE_FOLDER = "./data/clean/pkl_v2/"

def cps_to_wps_evan(cps, is_white):
    """
    Converts 2d array of centipawn scores to win prob 
    
    passed through to get_vocs() to calculate vocs in terms of win percentage
    """
    
    wps = np.empty(cps.shape)
    
    for i in range(cps.shape[0]):
        for j in range(cps.shape[1]):
            
            # if dealing w/ centipawns
            if np.abs(cps[i,j])< 10000:
                
                # note: cp_to_wp assumes framing in white, but cps is framed as active 
                wps[i,j] = cp_to_wp( (2*is_white-1)*cps[i,j] )
                
                # wps[i,j] = sigmoid(offset*(2*is_white-1) + k*cps[i,j])
                
            else:
                
                # if dealing with mate -
                # note that mates were stored w/ 10000 + mate_number to preserve same format as centipawns
                # mate_to_wp takes in from active player.
                wps[i,j] = mate_to_wp(cps[i,j] - np.sign(raw_score)*10000) # edited this
                
    return wps

def get_vocs(values_raw, wp_function, is_white, depth_0=0, depth_f=-1, fixed_sd = 0.0001, sdm = 1, upper=1,lower=0,step=0.001, is_eco = False):
    """
    Calculates expected voc from mean and standard deviations derived from values_cp array
    """
    values_cp = values_raw.copy()
    if is_eco:
        values_cp = values_cp[:5]
    
    # get values as win probabilities
    values = wp_function(values_cp, is_white)
    
    # get the means and variances of the distributions
    mu = values[:,depth_0]
    sigma = sdm * np.abs(values[:,depth_0]-values[:,depth_f])
    sigma = np.clip(sigma, fixed_sd, None)
    
   
    # use a product of cdfs trick to calculate the expectation of the max of several gaussian distributions
    x = np.arange(lower,upper,step)
    
    cdf = np.prod(stats.norm.cdf(np.tile(x,[len(mu),1]).T,loc=np.tile(mu,[len(x),1]),scale=np.tile(sigma,[len(x),1])),axis=1)
    cdf[-1]= 1
    
    max_ev = lower + np.trapz(1-cdf,dx=step)
    
    # calculate the expected value of the myopic choice 
    i_max = np.argmax(mu)
    
    cdf = stats.norm.cdf(x,loc=mu[i_max],scale=sigma[i_max])
    cdf[-1]= 1
    
    base_ev = lower + np.trapz(1-cdf,dx=step)
    
    # return the expectation of the best option after - before search
    return max_ev - base_ev        
    
def get_evan(values_cp, wp_function, is_white):
    """
    Calculates the realized voc from value_cp array
    """
    values = wp_function(values_cp, is_white)
    idx = np.argmax(values[:,0])
    return np.max(values[:,-1])-values[idx,-1]


class FILE_iter:
    """
    Iterates through each file of about 10,000 games
    """
    def __init__(self):      
        file_paths = []
        for dir_name in RAW_FOLDERS:
            files = os.listdir(dir_name)
            for file in files:
                file_paths.append(os.path.join(dir_name,file))
        self.files = file_paths
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if len(self.files) > 0:
            return self.files.pop()
        else:
            raise StopIteration
        


def process_file_path(file_path, time_ctrl, total_time, max_rt, lower_ply, upper_ply):
    """
    Main processing function. Reads the pickle file, selects relavent games, strips data
    and adds voc calculations.
    """
    
    # select relavent games
    df = pd.read_pickle(file_path)
    df = df[(df.rt > -1) & (df.rt < max_rt) & (df.time_control == time_ctrl)]
    df['remaining'] = df['num_ply'] - df['move_ply']
    df = df[(df.move_ply >= lower_ply) & (df.move_ply <= upper_ply) & (df.remaining > 2)]
 
    result = []
    
    # add VOC data
    if len(df) > 0:      
            df['voc'] = df.apply(lambda x: get_vocs(x['values'], cps_to_wps_evan, x['move_ply']%2==0, 
                                                    depth_0=0, depth_f=-1,step=0.005, is_eco=True), axis=1)
            df['evan'] = df.apply(lambda x: get_evan(x['values'], cps_to_wps_evan, x['move_ply']%2==0), axis=1)
            df['time-rem'] = df.apply(lambda x: int(total_time - int(x['clock'])), axis=1)

            result = df[['game_id','move_ply','time-rem','rt','elo','evan','voc']]

            
    return result
            
if __name__ == "__main__":
    """
    Multithreading processing of game files
    """

    n_cores = len(os.sched_getaffinity(0)) -6
    print('Starting v4')
    print('Number of cores: ' + str(n_cores))

    total_times = [60,120,180, 180, 300,300,
                   600, 600, 900, 1800, 1800]
    max_rts     = [30, 60, 90,90,120,120,
                   180,180,240,240, 240]
    time_ctrls = ['60+0','120+1','180+0','180+2','300+0','300+3',
                  '600+0','600+5', '900+10', '1800+0', '1800+20']
    
    lower_ply = 15
    upper_ply = 75

    for i, time_ctrl in enumerate(time_ctrls):

        total_time = total_times[i]
        max_rt = max_rts[i]
        
        def process_function(file_path):
            #print(file_path)
            return process_file_path(file_path, time_ctrl, total_time, max_rt, lower_ply, upper_ply)

        file_iter = FILE_iter()
  
        save_file = SAVE_FOLDER + time_ctrl.replace("+", "_" ) + "_full"

        n = 0
        results = []
        
        pool = Pool(n_cores)
        
        for result in pool.imap_unordered(process_function, file_iter):

            if len(result) > 2:
                results.append(result)
                n += 1

                #Partial save
                if n%50 == 49:
                    print(n)
                    sys.stdout.write
                    pd.concat(results, ignore_index=True).to_csv(save_file +".csv", index=False)
                    pd.concat(results, ignore_index=True).to_pickle(save_file +".pkl")


        #Final save             
        pd.concat(results, ignore_index=True).to_csv(save_file +".csv", index=False)
        pd.concat(results, ignore_index=True).to_pickle(save_file +".pkl")

        print("Final file save")
        sys.stdout.write
    
        pool.close()
        pool.terminate()
        pool.join()         
            
            

        

    

    
    
    
    
    
    
    
    
    
    
    
    
    
