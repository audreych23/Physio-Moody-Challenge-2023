import sys
from helper_code import *
from team_code import train_challenge_model
import graphing as plotter

if __name__ == '__main__':
    # Parse the arguments.
    if not (len(sys.argv) == 3):
        raise Exception('Include the data and graph folders as arguments, e.g., data_preview.py data model.')

    # Define the data and model foldes.
    data_folder = sys.argv[1]
    graph_folder = sys.argv[2]
    print('Finding Data Folder...')
    patient_ids = find_data_folders(data_folder)
    # Create a folder for the model if it does not already exist.
    os.makedirs(graph_folder, exist_ok=True)
    plotter.plot_data_graph(patient_ids, data_folder, graph_folder)
    print('Done.')