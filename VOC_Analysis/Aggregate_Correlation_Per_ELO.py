# This script computes mean spearman correlation (RT vs VOC)  per  ELO and Time condition for a given month of saved VOCs / RTs (split by job)

# LOAD PACKAGES
import os
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import time
import sys
import time
import pickle
import scipy.stats

# GET JOB INDICES (WHICH MONTH TO ANALYZE)
if is_array_job:
    job_idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
else: 
    job_idx = 0

# set up working folder
working_folder = '/home/erussek/projects/Chess_Project';
os.chdir(working_folder)

# make folder to save res
to_save_folder = os.path.join(working_folder, "aggregate_r_spear_vs_elo")
if not os.path.exists(to_save_folder):  
  os.makedirs(to_save_folder)

# find saved move data
data_folder_outer_scratch = "/scratch/gpfs/erussek/Chess_project"

data_folder_inners_scratch = ["analysis_results_Jan_2019_60+0_full_3_sf14/move_level",
  "analysis_results_Jan_2019_180+0_full_3_sf14/move_level_Jan180",
  "analysis_results_Jan_2019_300+0_full_3_sf14/move_level_Jan300",
  "analysis_results_Jan_2019_600+0_full_3_sf14/move_level",
  "move_level_Feb",
  "move_level_March",
  "analysis_results_April_2019_full_3_sf14/move_level_April",
  "analysis_results_May_2019_full_3_sf14/move_level",
  "analysis_results_June_2019_full_3_sf14/move_level",
  "analysis_results_July_2019_full_3_sf14/move_level",
  "analysis_results_Aug_2019_full_3_sf14/move_level_Aug"]


data_folders_scratch = [os.path.join(data_folder_outer_scratch,inner) for inner in data_folder_inners_scratch]

# place all directory folders in a list
data_folder = data_folders_scratch[job_idx]
directory_contents = os.listdir(data_folder)
print('N files: {}'.format(len(directory_contents))) 


def get_midpoints(x):
    return (x[1:] + x[:-1]) / 2

def pool_corr_means(res_df_old, res_df_new):
    
    res_df1 = res_df_old
    res_df2 = res_df_new
    
    res_df_c = res_df1.copy()
        
    res_df_c['r_spear_count'] = res_df1['r_spear_count']+res_df2['r_spear_count']
    res_df_c['r_spear_mean']= res_df1['r_spear_mean']*(res_df1['r_spear_count']/res_df_c['r_spear_count']) + res_df2['r_spear_mean']*(res_df2['r_spear_count']/res_df_c['r_spear_count'])
    res_df_c['r_spearsquared_mean']= res_df1['r_spearsquared_mean']*(res_df1['r_spear_count']/res_df_c['r_spear_count']) + res_df2['r_spearsquared_mean']*(res_df2['r_spear_count']/res_df_c['r_spear_count'])

    res_df_c.loc[res_df_c.r_spear_count == 0, 'r_spear_mean'] = 0
    res_df_c.loc[res_df_c.r_spear_count == 0, 'r_spearsquared_mean'] = 0
    
    return res_df_c

# TIME CONTROL SETTTINGS
these_tcs = ['60+0', '120+1', '180+0',
                '180+2', '300+0', '300+3',
                '600+0', '600+5', '900+10',
                '1800+0', '1800+20']

# ERROR IN ORIGINAL SCRIPT DIDN'T ADD BACK INCREMENT FOR +10 AND +20 SO ADD THIS BACK
add_back_error = [0,0,0,
                 0,0,0,
                 0,0,10,
                 0,20]

# STORE FOR EACH TIME-CONTROL SETTING IN A LIST
r_spear_dfs = []

start = time.time()

for i in range(len(directory_contents)):

    print(i, end= ' ')
    file_idx = i
    file_name =  directory_contents[file_idx]
    this_file = os.path.join(data_folder, file_name)
    full_move_df = pd.read_csv(this_file, index_col = 0)
    full_move_df =full_move_df.dropna(subset=['white_active']).reset_index(drop=True)
    full_move_df['active_player'] = full_move_df['black_player']
    full_move_df.loc[full_move_df.white_active, 'active_player'] = full_move_df['white_player']
    full_move_df['game_player'] = full_move_df['game_id'] + full_move_df['active_player']

    # FILTER TO MOVE PLYS OF INTEREST (BTW. 15 AND 75)
    full_move_df = full_move_df.loc[(full_move_df.move_ply >= 15) & (full_move_df.move_ply <= 75)].reset_index(drop=True)

    for tc_idx in range(len(these_tcs)):

        move_df = full_move_df.loc[full_move_df.time_control == these_tcs[tc_idx]].reset_index(drop=True)

        move_df['rt'] = move_df['rt'] + add_back_error[tc_idx]

        elo_bin_bounds = np.linspace(800, 2200, 8)
        elo_bin_bounds = np.append(0, elo_bin_bounds)
        elo_bin_bounds = np.append(elo_bin_bounds, 3000)
        elo_bin_labels = get_midpoints(elo_bin_bounds)
        move_df['elo_bin'] = pd.cut(move_df['active_elo'], elo_bin_bounds, labels=elo_bin_labels)

        if move_df.shape[0] == 0:
            elo_corr_agg = pd.DataFrame({'elo_bin': elo_bin_labels, 'r_spear_mean': len(elo_bin_labels)*[0], 'r_spear_count': len(elo_bin_labels)*[0], 'r_spearsquared_mean': len(elo_bin_labels)*[0]})
            elo_corr_agg = elo_corr_agg.set_index('elo_bin')
        else:  
            elo_corr_df = move_df.groupby(['elo_bin']).apply(lambda x: scipy.stats.spearmanr(x['voc'], x.rt)[0]).reset_index().rename(columns={0:"r_spear"})
            elo_corr_df['r_spearsquared'] = np.power(elo_corr_df['r_spear'],2)
            elo_corr_agg = elo_corr_df.groupby('elo_bin').agg({'r_spear': ['mean', 'count'], 'r_spearsquared': 'mean'})
            elo_corr_agg.columns= elo_corr_agg.columns.get_level_values(0)+ '_' + elo_corr_agg.columns.get_level_values(1)
            elo_corr_agg.loc[elo_corr_agg.r_spear_count == 0, 'r_spear_mean'] = 0
            elo_corr_agg.loc[elo_corr_agg.r_spear_count == 0, 'r_spearsquared_mean'] = 0

        if i == 0:

            r_spear_dfs.append(elo_corr_agg.copy())

        else:
            r_spear_dfs[tc_idx] = pool_corr_means(r_spear_dfs[tc_idx], elo_corr_agg)

end = time.time()
print("Time taken: {}".format(end-start))

# SAVE RESULTS

for i in range(len(these_tcs)):
    
    file_name = 'r_spear_vs_elo_{}_{}'.format(these_tcs[i],job_idx)
    r_spear_dfs[i].to_csv(os.path.join(to_save_folder, file_name))

    
    