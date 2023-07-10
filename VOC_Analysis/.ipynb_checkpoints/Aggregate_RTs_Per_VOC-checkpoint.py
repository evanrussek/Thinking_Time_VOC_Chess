# This script runs on cluster and (in parallel) computes mean RT for each VOC / ELO / Time Control Setting - For a Given Month's Worth of Data

# LOAD PACKAGES
import os
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import time
import sys
import time
import pickle

is_array_job = True

# GET JOB INDICES (WHICH MONTH TO ANALYZE)
if is_array_job:
    job_idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
else: 
    job_idx = 0

# set up working folder
working_folder = '/home/erussek/projects/Chess_Project';
os.chdir(working_folder)

# make folder to save results
to_save_folder = os.path.join(working_folder, "aggregate_voc_vs_rt")
if not os.path.exists(to_save_folder):  
  os.makedirs(to_save_folder)

# find saved move data (from script compute Compute_Move_Vocs.py)
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

# Script will loop through saved files of vocs/rts, compute mean of each and iteratively update a mean rt (pooling) using these functions
def get_midpoints(x):
    return (x[1:] + x[:-1]) / 2

def pool_rt_means(res_df_old, res_df_new):
    
    res_df1 = res_df_old
    res_df2 = res_df_new
    
    res_df_c = res_df1.copy()
    res_df_c['rtcount']= res_df1['rtcount']+res_df2['rtcount']
    res_df_c['rtmean']= res_df1['rtmean']*(res_df1['rtcount']/res_df_c['rtcount']) + res_df2['rtmean']*(res_df2['rtcount']/res_df_c['rtcount'])
    res_df_c['rt_squaredmean']= res_df1['rt_squaredmean']*(res_df1['rtcount']/res_df_c['rtcount']) + res_df2['rt_squaredmean']*(res_df2['rtcount']/res_df_c['rtcount'])

    res_df_c.loc[res_df_c.rtcount == 0, 'rtmean'] = 0
    res_df_c.loc[res_df_c.rtcount == 0, 'rt_squaredmean'] = 0
    
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
rt_mean_total_dfs = []
rt_mean_total_dfs_split_elo = []
n_total_elo_game_players = []
n_games = []

start = time.time()

# LOOP THROUGH FILES IN MONTH and ITERATIVELY UPDATE MEAN RTS PER VOC / TIME-CONTROL-SETTING
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

    # LOOP THROUGH TIME-CONTROL SETTINGS
    for tc_idx in range(len(these_tcs)):

        # GET MOVES FOR THIS TIME-CONTROL SETTING
        move_df = full_move_df.loc[full_move_df.time_control == these_tcs[tc_idx]].reset_index(drop=True)
        
        # ADD BACK TIME IF ERROR IN RT (SEE ABOVE)
        move_df['rt'] = move_df['rt'] + add_back_error[tc_idx]

        # BIN VOC
        n_voc_bin_bounds = 20
        voc_bin_bounds = np.linspace(-.01,.4,n_voc_bin_bounds)
        voc_bin_bounds = np.append(voc_bin_bounds, 1)
        voc_bin_labels = get_midpoints(voc_bin_bounds)
        move_df['voc_bin'] = pd.cut(move_df['voc'], voc_bin_bounds, labels=voc_bin_labels)

        # BIN ELO
        elo_bin_bounds = np.array([1250, 1525, 1800])
        elo_bin_bounds = np.append(0, elo_bin_bounds)
        elo_bin_bounds = np.append(elo_bin_bounds, 3000)
        elo_bin_labels = get_midpoints(elo_bin_bounds)
        move_df['elo_bin'] = pd.cut(move_df['active_elo'], elo_bin_bounds, labels=elo_bin_labels)

        # STORE RT_SQUARED TO COMPUTE ERROR BARS
        move_df['rt_squared'] = np.power(move_df['rt'],2)

        # GET MEAN RT (AND SQUARED) FOR EACH ELO BIN / VOC BIN
        res_df = move_df.groupby(['voc_bin', 'elo_bin']).agg({'rt': ['mean', 'count'], 'rt_squared': ['mean']}).reset_index() 
        res_df.columns= res_df.columns.get_level_values(0)+res_df.columns.get_level_values(1)

        # REPLACE count=0 with 0 to not screw up iterative adding
        res_df.loc[res_df.rtcount == 0, 'rtmean'] = 0
        res_df.loc[res_df.rtcount == 0, 'rt_squaredmean'] = 0
        
        # ALSO RUN  w/o SPLITTING ELO
        res_df_no_elo = move_df.groupby(['voc_bin']).agg({'rt': ['mean', 'count'], 'rt_squared': ['mean']}).reset_index() 
        res_df_no_elo.columns= res_df_no_elo.columns.get_level_values(0)+res_df_no_elo.columns.get_level_values(1)
        res_df_no_elo.loc[res_df_no_elo.rtcount == 0, 'rtmean'] = 0
        res_df_no_elo.loc[res_df_no_elo.rtcount == 0, 'rt_squaredmean'] = 0
        
        # HANDLE POOLING
        
        if move_df.shape[0] > 0:
            n_games_df = move_df.groupby('elo_bin').agg({'game_player': 'nunique', 'game_id': 'nunique'}).reset_index()
        else:
            n_games_df = pd.DataFrame({'elo_bin': elo_bin_labels})
            n_games_df['game_player']=0
            n_games_df['game_id']=0
            
        if i == 0:
            rt_mean_total_dfs_split_elo.append(res_df.copy())
            rt_mean_total_dfs.append(res_df_no_elo.copy())

            n_total_elo_game_players.append(n_games_df)
            n_games.append(move_df.game_id.nunique())
        else:
            rt_mean_total_dfs_split_elo[tc_idx] = pool_rt_means(rt_mean_total_dfs_split_elo[tc_idx], res_df)
            rt_mean_total_dfs[tc_idx] = pool_rt_means(rt_mean_total_dfs[tc_idx], res_df_no_elo)

            n_total_elo_game_players[tc_idx]['game_player'] = n_total_elo_game_players[tc_idx]['game_player'] + n_games_df['game_player']
            n_total_elo_game_players[tc_idx]['game_id'] = n_total_elo_game_players[tc_idx]['game_id'] + n_games_df['game_id']
            n_games[tc_idx] = n_games[tc_idx] + move_df.game_id.nunique()

end = time.time()
print("Time taken: {}".format(end-start))

# SAVE EVERYTHING
for i in range(len(these_tcs)):
    
    file_name = 'voc_vs_RT_{}_{}'.format(these_tcs[i],job_idx)
    rt_mean_total_dfs[i].to_csv(os.path.join(to_save_folder, file_name))
    file_name = 'voc_vs_RT_split_elo_{}_{}'.format(these_tcs[i],job_idx)
    rt_mean_total_dfs_split_elo[i].to_csv(os.path.join(to_save_folder, file_name))
    file_name = 'n_total_elo_game_players_{}_{}'.format(these_tcs[i],job_idx)
    n_total_elo_game_players[i].to_csv(os.path.join(to_save_folder, file_name))

    
file_name = 'n_games_total_{}'.format(job_idx)
with open(os.path.join(to_save_folder, file_name), "wb") as fp:   #Pickling
    pickle.dump(n_games, fp)
    
    