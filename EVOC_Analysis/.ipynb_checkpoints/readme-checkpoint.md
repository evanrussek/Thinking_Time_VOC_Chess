This code generates Figs 4 and 5 from the manuscript.

This document provides instructions on how to run the software consisting of three programs: chess_csv.py, create_pkl.py, and create_summary.py. These programs process chess move data in order to calculate expected value of computation (EVOC) and relate them to move times and player strength. Please follow the steps outlined below to set up and execute each program.


Program 1: chess_csv.py

chess_csv.py is a program that extracts relevant move data, including Stockfish values, from a database CSV file.

Prerequisites
	Download stockfish_14_x64 from stockfishchess.org.
	Set the variable ENGINE_FILE in chess_csv.py to the file location of stockfish_14_x64.
	Download a Lichess database CSV file.
	Set the variable CSV_FILE in chess_csv.py to the file location of the Lichess database CSV.
	Set the variable TARGET_FILE in chess_csv.py to the folder where you want to save the results.
Execution
	Ensure you have completed the prerequisites mentioned above.
	Run the command python chess_csv.py.


	
Program 2: create_pkl.py

create_pkl.py is a program that combines raw files generated by chess_csv.py into a large .pkl file containing EVOC (expected value of computation) data.

Prerequisites
	Set the variable RAW_FOLDERS in create_pkl.py to the directories storing the raw files generated by chess_csv.py.
	Set the variable SAVE_FOLDER in create_pkl.py to the desired file destination for the generated .pkl file.
Execution
	Ensure you have completed the prerequisites mentioned above.
	Run the command python create_pkl.py.

	
Program 3: create_summary.py

create_summary.py is a program that generates EVOC summary data.

Prerequisites
	Set the variable SOURCE_FOLDER in create_summary.py to the folder containing the required data generated by create_pkl.py.
	Set the variable SAVE_FILE in create_summary.py to the desired file destination for the generated summary data.
Execution
	Run the command python create_summary.py.
