import os
import glob
import sys
from helper_code import *


def modify_recording_metadata(directory_data_folder, threshold, id):
    input_recording_metadata_file = os.path.join(directory_data_folder, id + '.tsv')
    # To overwrite old metadata
    output_recording_metadata_file = os.path.join(directory_data_folder, id + '.tsv')
    input_recording_metadata = load_text_file(input_recording_metadata_file)

    hours = get_hours(input_recording_metadata)
    indices = [i for i, hour in enumerate(hours) if hour >= threshold]
    
    input_lines = input_recording_metadata.split('\n')
    lines = [input_lines[0]] + [input_lines[i + 1] for i in indices]
    output_recording_metadata = '\n'.join(lines)
    with open(output_recording_metadata_file, 'w') as f:
        f.write(output_recording_metadata)
    return

def delete_files_with_pattern(folder_path, pattern):
    # Combine the folder path and pattern to form the complete search path
    search_path = os.path.join(folder_path, pattern)

    # Use glob to find files matching the pattern
    files_to_delete = glob.glob(search_path)

    # Delete each file in the list
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to delete: {file_path}. Error: {e}")

def generate_pattern(threshold):
    """Generate pattern to be deleted below a certain threshold
    Args:
        threshold: a number less than 72
    Returns:
        patterns to be deleted depending on threshold
    """
    patterns = list()
    for i in range(1, threshold):
        if i > 0 and i < 10:
            id = "0" + f"{i}" 
        else:
            id = f"{i}"
        pattern = f"*_*_{id}.*"
        patterns.append(pattern)

    return patterns

if __name__ == "__main__":
    if not (len(sys.argv) == 3):
        raise Exception('Include the data folders, threshold as arguments, e.g., python delete_signal_below_threshold.py data threshold.')
    data_folder = sys.argv[1]
    threshold = int(sys.argv[2])
    # Generate patterns to be deleted
    patterns = generate_pattern(threshold)
    # List all patient directories
    for directory in os.listdir(data_folder):
        directory_data_folder = os.path.join(data_folder, directory)
        # Modify the metadata
        modify_recording_metadata(directory_data_folder, threshold, directory)
        # Delete the files containing the same pattern for every patient
        for pattern in patterns:
            delete_files_with_pattern(directory_data_folder, pattern)