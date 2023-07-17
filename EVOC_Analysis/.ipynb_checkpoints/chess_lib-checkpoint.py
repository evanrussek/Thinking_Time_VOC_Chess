# This library is used to process games and evaluate stockfish valuations at multiple depths
# 

#load packages
import numpy as np
import pandas as pd
import chess
import stockfish
from stockfish import Stockfish
import chess.engine
import chess.pgn

#engine = chess.engine.SimpleEngine.popen_uci("./stockfish_14_1")
#engine.configure({'Threads':1,'Use NNUE':True, "Hash": 32})
#stockfish = Stockfish("./stockfish_14_1")

def get_cp(score, ply = 60):
    """
    Gets centipawn score or mate value from chess score object
    """
    
    cp = score.relative.score()
    mate = score.relative.mate()
    
    if cp is not None:
        return min(3000, max(cp,-3000))
    elif mate is not None:
        return mate + np.sign(mate)*10000

def get_consideration_set(engine, fen, multipv, max_depth):
    """
    Generates the set of moves considered to be used in the VOC calculation
    """
    
    moves = []
    
    board = chess.Board(fen)
    if board.is_game_over(): return moves

    engine.configure({"Clear Hash": 1})
    analysis_0 = engine.analyse(board, chess.engine.Limit(depth=0), multipv=multipv)
    
    for pv in range(len(analysis_0)):
        move = analysis_0[pv]['pv'][0]
        if move not in moves:
            moves.append(move)
    
    for d in range(max_depth):
        
        engine.configure({"Clear Hash": 1})
        analysis = engine.analyse(board, chess.engine.Limit(depth=d))
        move = analysis['pv'][0]
        
        if move not in moves:
            moves.append(move)
            
    return moves


def get_values(engine, fen, multipv = 5, depths=[1,5,10] ):
    """
    Uses the stockfish engine passed in to compute the centipawn values of the 
    moves in the consideration set. 

    Returns: a 2d array of values and the corresponding moves.
    """
    
    # set the type of value calculated
    calc_value = get_cp
    inv_value = 0
    
    max_depth = depths[-1]
    
    # load game
    board = chess.Board(fen)
    ply = board.ply()
    
    # get consideration set and initialize values.
    moves = get_consideration_set(engine, fen, multipv, max_depth)
    values = -20000*np.ones([len(moves), len(depths) + 1])
     
    # get all depth 0 values via large multipv depth0    
    engine.configure({"Clear Hash": 1})
    engine.configure({"SyzygyProbeLimit": 0})
    analysis_0 = engine.analyse(board, chess.engine.Limit(depth=0), multipv=100)
    engine.configure({"SyzygyProbeLimit": 7})
    
    for pv in range(len(analysis_0)):
        
        move = analysis_0[pv]['pv'][0]
        if move in moves:
            i = moves.index(move)
            values[i,0] = calc_value(analysis_0[pv]['score'], ply)
    
    #Assign all other depth values
    for i, move in enumerate(moves):
    
        board = chess.Board(fen)
        board.push(move)
        
        if board.is_game_over():
            values[i,1:] = values[i,0]
        
        for j, depth in enumerate(depths):     
            engine.configure({"Clear Hash": 1})
            analysis = engine.analyse(board, chess.engine.Limit(depth=max(0,depth-1)))
            values[i,j+1] = inv_value - calc_value(analysis['score'], ply)

        
    return values, moves
 
def get_value_evan(values):
    """
    Get the simple VOC values often refered to as evan value.
    """
    try:
        idx = np.argmax(values[:,0])
        return np.max(values[:,-1])-values[idx,-1]
    except:
        return np.nan

    

def get_deltas(engine, fen, mpv, depths):
    """
    Gets the change in values over the course of increasing search depth
    Used for illustrative purposes
    """
    
    values, moves = get_values(engine, fen, mpv, depths)
    
    fens = []
    deltas = np.zeros([len(moves), len(depths)])
    
    for i in range(len(moves)):
        
        board = chess.Board(fen)
        board.push(moves[i])
        
        fens.append(board.fen())
        
        deltas[i] = values[i, 1:] - values[i,0]
    
    return deltas, fens