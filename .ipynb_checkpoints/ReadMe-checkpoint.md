# Code for "Time spent thinking in online chess reflects the value of computation"

Preprint: https://psyarxiv.com/8j9zx/

Contact Evan Russek (evrussek@gmail.com) or Dan Acosta-Kane (dan.acostakane@gmail.com) with questions

## Folder Structure 

- VOC_Analysis examines relationship between value of computation and response time (Figs. 1, 2, and 3).

- EVOC_Analsis defines the expected value of computation and tests relationships between this and response time (Figs. 4 and 5).

- Cost_of_Time_Analysis empirically measures the cost of spending time and computes the implied optimal move times based on this (Figs. 6 and 7)

Each folder contains its own specification for how to reproduce analysis.

## Requirements

Data: All data comes from the lichess.org database. We specifically analyzed a subset of this data, utilized for the Maia Chess Project, which is formated into CSVs and available here: http://csslab.cs.toronto.edu/datasets/#monthly_chess_csv. We analyzed the files January through August.

Engine: Analysis uses the stockfish chess engine https://stockfishchess.org/, version 14. This can be downloaded at https://www.dropbox.com/sh/75gzfgu7qo94pvh/AAB3yfXx6PkUfHbBr4r_RwhGa.

We interface with the engine using the python chess package, version 1.9.3 https://github.com/niklasf/python-chess.