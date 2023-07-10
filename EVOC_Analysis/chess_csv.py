# This script generates and computes individual move data from database .csv file
# 
# Individual games are cleaned, processed, and move values in centipawn scores are
# added. Results are saved to pickle files covering 10,000 games each. Columns of 
# the pickle are: game_id, time_control, clock, num_ply, move_ply, move, board, elo,
# elo_op, win, rt, values, move_value.



# load packages
import numpy as np
import pandas as pd
import chess_lib
from chess_lib import process_game, process_game_win, process_game_var
import os
import sys
from multiprocessing import Pool
import chess.pgn
from multiprocessing import util
import threading
import copy
from datetime import date
import csv
import time


sys.setrecursionlimit(10000)

# file locations and desitions
ENGINE_FILE ="./stockfish_14_x64"
CSV_FILE = "/persistent/ChessRT/games/csv/lichess_db_standard_rated_2019-04.csv"
TARGET_FILE = 'data/raw/csv_apr/all'

# csv column names and desired columns
ALL_COLUMNS = ['game_id', 'type', 'result', 'white_player', 'black_player', 'white_elo', 'black_elo', 'time_control', 'num_ply', 'termination', 'white_won', 'black_won', 'no_winner', 'move_ply', 'move', 'cp', 'cp_rel', 'cp_loss', 'is_blunder_cp', 'winrate', 'winrate_elo', 'winrate_loss', 'is_blunder_wr', 'opp_winrate', 'white_active', 'active_elo', 'opponent_elo', 'active_won', 'is_capture', 'clock', 'opp_clock', 'clock_percent', 'opp_clock_percent', 'low_time', 'board', 'active_bishop_count', 'active_knight_count', 'active_pawn_count', 'active_queen_count', 'active_rook_count', 'is_check', 'num_legal_moves', 'opp_bishop_count', 'opp_knight_count', 'opp_pawn_count', 'opp_queen_count', 'opp_rook_count']
COLUMNS = ['game_id','time_control', 'clock', 'opp_clock','white_elo','black_elo','num_ply', 'white_won', 'move_ply', 'move' ,'white_active', 'board']

# select game controls of interest
GAME_CTRL =  ['60+0', '120+1', '180+0','180+2', '300+0', '300+3', '600+0', '600+5', '900+10', '1800+0', '1800+20']
COL_DTYPE = {'game_id': 'string', 'time_control': 'string', 'clock':'int16', 'white_elo':'int16', 'black_elo':'int16', 
                'num_ply':'int16', 'white_won':'string', 'move_ply':'int16', 'move':'string', 'white_active':'string', 'board':'string'}

# start where we last left off
start_number = 0
for file in os.listdir('data/raw/csv_apr'):
    n = int(file[15:22])
    
    if n > start_number:
        start_number = n

NS = np.arange(start_number,5000000,10000)

class CSV_iter:
    """
    Create game iterater to facilitate multithreading
    Returns a dataframe of the relavent move wise data
    Selects only for desired time control settings
    """
    def __init__(self, filename, start, stop):
        self.filename = filename
        f = open(filename)
        self.reader = csv.reader(f)
        self.i = 0
        self.stop = stop
        self.columns = next(self.reader)
        self.next_row = next(self.reader)
        self.prev_row = ''
        
        self.skip(start)
        
    def __iter__(self):
        return self
    
    def skip(self, n):
        if n == 0: 
            return
    
        for row in self.reader:
            self.i+=1
            if self.i >= n:
                return 
    
    def get_game(self):
        rows = []
        first_row = self.next_row
        game_id = self.next_row[0]
        
        self.i+=1
        for row in self.reader:     
            rows.append(self.next_row)
            self.prev_row = self.next_row
            self.next_row = row
            if row[0] != game_id:
                break
                
        df = pd.DataFrame(rows, columns=self.columns)
        
        if df['time_control'][0] not in GAME_CTRL:
            return self.get_game()
            
        df = df[COLUMNS]
        df = df.astype(COL_DTYPE)
            
        return df
            
        
    
    def __next__(self):
        if self.i < self.stop:
            try:
                return self.get_game()
            except: 
                raise StopIteration
        else:
            raise StopIteration

def clean_game(df):
    """
    Ensures that game dataframe has proper data
    """
    
    assert len(pd.unique(df.game_id)) == 1
    
    if 'white_elo' not in df:
        print('return')
        return df
    
    df['white_active'] = (df.white_active == 'True')
    df['white_won'] = (df.white_won == 'True')
    df['elo']    = df.white_elo*df.white_active + df.black_elo*(1-df.white_active)
    df['elo_op'] = df.white_elo*(1-df.white_active) + df.black_elo*(df.white_active)
    df['win']    = df.white_won*df.white_active + (1-df.white_won)*(1-df.white_active)
    
    
    
    time_control = df['time_control'][0]
    start_time = int(time_control.split('+')[0])
    add_time = int(time_control.split('+')[1])
    
    white_rt = add_time - df.loc[df['white_active']==True,'clock'].diff()
    black_rt = add_time - df.loc[df['white_active']==False,'clock'].diff()
    
    df.loc[df.white_active==True,'rt']  = white_rt
    df.loc[df.white_active==False,'rt'] = black_rt
    
    df = df.drop(['white_elo', 'black_elo','opp_clock','white_active','white_won'], axis=1)
    
    return df

 
def add_values(df, engine_file= ENGINE_FILE,mpv=5,depth=[1,15]):
    """
    Adds movewise stockfish values to the game dataframe
    Stockfish values are in centipawns and are generated by chess_lib.get_values()
    They are stored as a 2D array. Each row is for each of the mpv=5 moves 
    considered. And for each move multiple depths are evaluated (depth=[1,15])
    """
    
    engine = chess.engine.SimpleEngine.popen_uci(engine_file, timeout=300)
    engine.configure({'Threads':1,'Use NNUE':True, "Hash": 16})
    
    df['values'] = np.nan
    df['values'] = df['values'].astype(object)
    df['move_value'] = np.nan
    
    for i, fen in enumerate(df.board):
        
        # try except to catch occasions when the engine crashes
        try:
            values, moves = chess_lib.get_values(engine, fen, mpv, depth)
            move_value = add_move_value(fen, df.at[i,'move'], engine, depth=depth[-1])

        except Exception as e:
            print(repr(e))
            engine.quit()
            engine = chess.engine.SimpleEngine.popen_uci(engine_file)
            engine.configure({'Threads':1,'Use NNUE':True, "Hash": 16})

            try:
                values, moves = chess_lib.get_values(engine, fen, mpv, depth)
                move_value = add_move_value(fen, df.at[i,'move'], engine, depth=depth[-1])
            except:
                engine.quit()
                return df
        
        
        df.at[i,'values'] = np.int16(values)
        df.at[i,'move_value'] = np.int16(move_value)
        
    engine.quit()
    
    
    return df


def add_move_value(fen, move, engine, depth=15):
    """
    Adds stockfish value of the actual move selected
    """
    
    board = chess.Board(fen)
    ply = board.ply()
    
    board.push(chess.Move.from_uci(move))
    engine.configure({"Clear Hash": 1})
    analysis = engine.analyse(board, chess.engine.Limit(depth=max(0,depth-1)))
    
    return -chess_lib.get_cp(analysis['score'], ply)

def process_function(game_df):
    """
    Creates process function to be passed to multithreader
    """
    game_df = clean_game(game_df)
    game_df = add_values(game_df, engine_file=ENGINE_FILE, mpv=5,depth=[1,15])
    return game_df


def process_game_wrapper(game): #Old process function wrapper no longer used.
    return process_game(game, engine_file=ENGINE_FILE, mpv=5, depths=[1,2,3,4,5,10,15])


if __name__ == "__main__":
    """
    Main script takes csv file and processes the game data resulting in 
    multiple pickle files.
    """
    
    file_start = TARGET_FILE + '_' + date.today().isoformat() + '_'

    n_cores = len(os.sched_getaffinity(0)) -6
    print('Number of cores: ' + str(n_cores))
    pool = Pool(n_cores)

    ns = NS

    for i in range(1,len(ns)):
        n_0 = ns[i-1]
        n_f = ns[i]

        csv_iter = CSV_iter(CSV_FILE,n_0,n_f)

        results = []

        print("Start: " + str(n_0).zfill(7) + " end: " + str(n_f).zfill(7))
        sys.stdout.write

        filename = file_start + str(n_0).zfill(7) + '-' +str(n_f).zfill(7) + '.pkl'

        n = 0
        for result in pool.imap_unordered(process_function, csv_iter):

            if len(result) > 2:
                results.append(result)
                n += 1

                #Partial save
                if n%500 == 499:
                    print(n)
                    sys.stdout.write
                    pd.concat(results).to_pickle(filename)


        #Final save             
        pd.concat(results).to_pickle(filename)

        print("Final file save")
        sys.stdout.write
    
    pool.close()
    pool.terminate()
    pool.join()

    #sys.exit()
    
