import sys
from helper_code import *
import graphing as plotter
import os

os.environ['CUDA_VISIBLE_DEVICES'] ="0"

def write_csv(patient_ids, output_file):
    output_string = '\n'.join([str(elem) for elem in patient_ids])
    with open(output_file, 'w') as f:
        f.write(output_string)
    return

if __name__ == '__main__':
    # Parse the arguments.
    if not (len(sys.argv) == 4):
        raise Exception('Include the data and graph folders as arguments, e.g., data_preview.py data graph data.csv.')

    # Define the data and model foldes.
    data_folder = sys.argv[1]
    graph_folder = sys.argv[2]
    output_file = sys.argv[3]
    print('Finding Data Folder...')
    patient_ids = find_data_folders(data_folder)
    
    # Create a folder for the model if it does not already exist.
    os.makedirs(graph_folder, exist_ok=True)
    write_csv(patient_ids, output_file)
    plotter.plot_data_graph(patient_ids, data_folder, graph_folder)
    print('Done.')