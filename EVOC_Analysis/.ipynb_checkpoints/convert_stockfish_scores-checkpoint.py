# for cp to wp
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cp_to_wp(score_val_cp_white):
    """
    Converts centipawn score val (signed with white pos/black neg) to win prob for white
    
    Args:
        score_val_cp_white: centipawn value (signed with white positive black negative) for position
    Returns:
        win probability: measure between 0 and 1 giving probability of white win
        Note: taken for logistic regression model applied to predict win given centipawn games
    """
    
    # these values were discovered by logistic regression mapping SF cp evals to win-rates
    this_const = -0.045821224987387915;
    this_board_score_val = 0.002226678310106879

    return sigmoid(this_const + this_board_score_val*score_val_cp_white)

def mate_to_wp(score_val_mate):
    """
    Converts plys from mate to win prob (for the player who is within reach of mate)
    """
    # these values were discovered by logistic regression mapping SF mate evals to win-rates
    return sigmoid(3.6986223572286208 + -0.05930670977061789*np.abs(score_val_mate))


def process_score(this_score, white_active):
    """
    this score is what is returned from stockfish
    """
    
    white_score = this_score.white()
    is_mate = white_score.is_mate()
    
    if is_mate:
        score_type = 'mate'
        score_val = white_score.mate() # this is framed as white'
        if score_val > 0:
            wp = mate_to_wp(score_val) # this is framed as whichever side is close to mate
        elif score_val < 0:
            wp = 1 - mate_to_wp(score_val)
        elif score_val == 0:
            wp = 1 if white_active else 0
            # 1 or -1 depending on whose move this is.... so process score needs to take in who is active
    else: # centipawn...
        score_type = 'cp'
        score_val = white_score.score()
        wp = cp_to_wp(score_val)
        
    score_dict = {'type': score_type, 'val': score_val, 'wp': wp}
    return score_dict

def process_score_f2(cps, is_white):
    """
    Converts 2d array of centipawn scores (or mate) to win prob 
    
    This is the same as the above function, however deals with slightly different input data format (though functionally equivalent) which was used for EVOC analysis.
    
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