This code is used for figures 6 and 7.

MakeFigure6.ipynb plots qualitative predictions of optimal policies corresponding to different cost-structures.

Aggregate_Wins_By_WP_TL.py computes a value function (win rate for each combination of time-left and board position advantage, separately for each time-contol setting. It works from the results of Compute_Move_VOCs, which computes position values as well as VOC. It is run in parallel, with one job for each month,  saves the results in Saved_Quantities/aggregate_win_rates.

MakeFigure7.ipynb uses the results of this to make plots in figure 7. It plots the value function, computes the implied cost of time given these value functions, compute the optimal policy over move times given value functions as well as the mean time spent per move.