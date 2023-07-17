This code generates Figures 1, 2 and 3 from the manuscript.

MakeFigure1.ipynb generates Figure 1 which provides a demonstration of computation of VOC. It must be linked to Saved_Quantities/example_SF_moves_by_depth to be run. Those files contain stockfish move selections at each depth (1 - 16; the consideration set) for a set of example positions and evaluations of those moves at a high depth (depth 15).

Compute_Move_VOCs.py takes in a large csv of moves (path specified in the script - one of the month-specific files from http://csslab.cs.toronto.edu/datasets/#monthly_chess_csv). It then computes the value of computation and also response time for a subset of moves in the file. Note that this script is designed to be run over many jobs, in parallel, with each job computing VOCs for a subset of rows in the data csv file, and saving its results as a file. An example of how this script was called from the Princeton university research computing clusters is provided in Compute_Move_VOCs.slurm. This script also references convert_stockfish_scores.py which maps the output of stockfish evaluations (in units of centipawn and distance from mate) to units of time-agnostic win probability (referred to as U in the manuscript).

Aggregate_RTs_Per_VOC.py aggregates the results from the above to compute mean RT for each VOC / ELO / Time Control Setting, looping through files saved by Compute_Move_VOCspy. This was written to be run in parallel, with each job looking at files for a given month of data and saving in Saved_Quantities/aggregate_voc_vs_rt.

MakeFigure2.ipynb plots the results of this analysis and also generates tables related to number of games and moves. This must be linked to Saved_Quantities/aggregate_voc_vs_rt to be run, which stores results of aggregated rts per VOC / Time-control.

Aggregate_Correlation_Per_ELO.py computes the correlation of VOC with RT separately for different Elo bins. As above, this loops through files saved by Compute_Move_VOCs.py and is built to be run in parallel by month.

MakeFigure3.ipynb plots the results of this analysis (stored in Saved_Quantities/aggregate_r_spear_vs_elo)