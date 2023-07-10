#### This script computes, per time-control setting, the aggregate win rate conditioned on encountering a game-state with time-left T and position utility, U

# Built to be run in parallel, with 1 run for each month

# LOAD PACKAGES
import os
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import time
import sys
import time
import pickle
from convert_stockfish_scores import *

is_array_job = True

# GET JOB INDICES (WHICH MONTH TO ANALYZE)
if is_array_job:
    job_idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
else: 
    job_idx = 0

# set up working folder
working_folder = '/home/erussek/projects/Thinking_Time_VOC_Chess';
os.chdir(working_folder)

# make folder to save res
to_save_folder = os.path.join(working_folder, "aggregate_win_rates")
if not os.path.exists(to_save_folder):  
  os.makedirs(to_save_folder)

# find saved move data (from script compute Compute_Move_Vocs.py)
data_folder_outer_scratch = "/scratch/gpfs/erussek/Chess_project"

data_folder_inners_scratch = [ 
                      "analysis_results_Jan_2019_60+0_full_3_sf14/move_level",
                      "analysis_results_Jan_2019_180+0_full_3_sf14/move_level",
                      "analysis_results_Jan_2019_300+0_full_3_sf14/move_level",
                      "analysis_results_Jan_2019_600+0_full_3_sf14/move_level",
                        "move_level_Feb",
                        "move_level_March",
                        "analysis_results_April_2019_full_3_sf14/move_level_April",
                         "analysis_results_May_2019_full_3_sf14/move_level",
                      "analysis_results_June_2019_full_3_sf14/move_level",
                      "analysis_results_July_2019_full_3_sf14/move_level",
                          "analysis_results_Aug_2019_full_3_sf14/move_level", 
                     ]


data_folders_scratch = [os.path.join(data_folder_outer_scratch,inner) for inner in data_folder_inners_scratch]

# place all directory folders in a list
data_folder = data_folders_scratch[job_idx]
directory_contents = os.listdir(data_folder)
print('N files: {}'.format(len(directory_contents))) # this is between 0 and 5000

these_tcs = ['60+0', '120+1', '180+0',
                '180+2', '300+0', '300+3',
                '600+0', '600+5', '900+10',
                '1800+0', '1800+20']

these_maxts = np.array([60,120,180,
               180,300,300,
               600,600,900,
               1800,1800])

these_ntimes = these_maxts/3 + 1

def get_midpoints(x):
    return (x[1:] + x[:-1]) / 2

# process SF scores for current board position
def add_score(full_move_df):

    full_move_df['wp'] = 0 # of active... or white... 
    full_move_df['wp'] = full_move_df['wp'].mask((full_move_df.board_score_type == 'mate') & (full_move_df.board_score_val > 0), mate_to_wp(full_move_df.loc[(full_move_df.board_score_type == 'mate') & (full_move_df.board_score_val > 0), 'board_score_val']))
    full_move_df['wp'] = full_move_df['wp'].mask((full_move_df.board_score_type == 'mate') & (full_move_df.board_score_val < 0), 1 - mate_to_wp(full_move_df.loc[(full_move_df.board_score_type == 'mate') & (full_move_df.board_score_val > 0), 'board_score_val']))
    full_move_df['wp'] = full_move_df['wp'].mask((full_move_df.board_score_type == 'mate') & (full_move_df.board_score_val == 0) & (full_move_df.white_active == True), 1)
    full_move_df['wp'] = full_move_df['wp'].mask((full_move_df.board_score_type == 'mate') & (full_move_df.board_score_val == 0) & (full_move_df.white_active == False), 0) 
    full_move_df['wp'] = full_move_df['wp'].mask((full_move_df.board_score_type == 'cp'), cp_to_wp(full_move_df.loc[full_move_df.board_score_type == 'cp', 'board_score_val']))
    full_move_df['wp_active'] = full_move_df['wp'].mask((full_move_df.white_active==False), 1 - full_move_df['wp'])

    return full_move_df

val_dfs = []
start = time.time()

# LOOP THROUGH FILES IN MONTH and ITERATIVELY UPDATE WIN RATES Per T, U and TIME-CONTROL-SETTING
for i in range(len(directory_contents)):
    
    print(i, end= ' ')

    file_idx = i
    file_name =  directory_contents[file_idx]
    this_file = os.path.join(data_folder, file_name)
    
    try:
        full_move_df = pd.read_csv(this_file, index_col = 0, dtype = {'white_active': bool, 'white_won': bool})
        full_move_df =full_move_df.dropna(subset=['white_active']).reset_index(drop=True)
        full_move_df =full_move_df.dropna(subset=['board_score_type']).reset_index(drop=True)
        full_move_df =full_move_df.dropna(subset=['board_score_val']).reset_index(drop=True)

        full_move_df['active_player'] = full_move_df['black_player']
        full_move_df.loc[full_move_df.white_active, 'active_player'] = full_move_df['white_player']
        full_move_df['game_player'] = full_move_df['game_id'] + full_move_df['active_player']
        full_move_df = add_score(full_move_df)

        for tc_idx in range(len(these_tcs)):
            move_df = full_move_df.loc[full_move_df.time_control == these_tcs[tc_idx]].reset_index(drop=True)


            move_df['active_won'] = ((move_df['white_won'] & move_df['white_active']) | (move_df['black_won'] & (~move_df['white_active'])))

            filt_tc_df = move_df.loc[(move_df['opp_clock']>min(these_maxts[tc_idx]/3, 60))].reset_index(drop=True)

            wp_bin_bounds = np.linspace(-.0005,1.0005,num=50)
            wp_bin_labels = get_midpoints(wp_bin_bounds)

            tl_bin_bounds = np.linspace(0,these_maxts[tc_idx],num=int(these_ntimes[tc_idx]))
            tl_bin_labels=get_midpoints(tl_bin_bounds)

            filt_tc_df['wp_bin']=pd.cut(filt_tc_df['wp_active'], wp_bin_bounds, labels=wp_bin_labels)
            filt_tc_df['tl_bin']=pd.cut(filt_tc_df['clock'], tl_bin_bounds, labels=tl_bin_labels)

            filt_tc_df_sel = filt_tc_df[['wp_bin', 'tl_bin', 'active_won']]
            
            res_df = filt_tc_df_sel.groupby(['wp_bin', 'tl_bin']).agg({'active_won': ['sum', 'count']}).reset_index()
            res_df.columns= res_df.columns.get_level_values(0)+res_df.columns.get_level_values(1)

            if i == 0:
                val_dfs.append(res_df.copy())
            else:
                val_dfs[tc_idx]['active_wonsum']=val_dfs[tc_idx]['active_wonsum']+res_df['active_wonsum']
                val_dfs[tc_idx]['active_woncount']=val_dfs[tc_idx]['active_woncount']+res_df['active_woncount']
    except:
        print("failed to read data")
        
end = time.time()
print("Time taken: {}".format(end-start))

# save the results
for i in range(len(these_tcs)):
    file_name = 'emp_val_{}_{}'.format(these_tcs[i],job_idx)
    val_dfs[i].to_csv(os.path.join(to_save_folder, file_name))
