import os
import sys
import shutil

# This file is used to delete recorded file depending on highest_recorded hours below a certain threshold and is gotten from records
threshold = 46

if __name__ == '__main__':
    # Parse the arguments.
    if not (len(sys.argv) == 2):
        raise Exception('Include the data,folders as arguments, e.g., python clean_data_train.py model data outputs.')

    # Define the data folder
    data_folder = sys.argv[1]

    # Get all directories in the data folder
    directories = os.listdir(data_folder)

    for dir_name in directories:
        if not os.path.isfile(dir_name):
            new_path = os.path.join(data_folder, dir_name)
            # Change directory to the new folder
            os.chdir(new_path)

            # Check if RECORDS file exists
            if os.path.exists(os.path.join(new_path, 'RECORDS')):
                record_path = os.path.join(new_path, 'RECORDS')
                
                lst_records = []
                with open(record_path) as record_handler:
                    # Read the whole file
                    str_file = record_handler.read()
                    # Split the read file by newline
                    lst_records = str_file.splitlines()
                    
                # Get the final record
                final_record = lst_records[-1]
                # Parse the String : format will be ICARE_*_22
                # Only last part is needed
                highest_recorded_hours = int(final_record.split('_')[-1])
                print(f"patient id: {dir_name}, highest recorded hours: ", highest_recorded_hours)
                if highest_recorded_hours < threshold:
                    path_to_delete = os.path.join(data_folder, dir_name)
                    # print(path_to_delete)
                    print(f"Highest_recorded hours is {highest_recorded_hours} and is below the threshold: {threshold}. Removing the '{path_to_delete}' directory tree permanently..")
                    shutil.rmtree(path_to_delete)
            else:
                raise('No Record File Exist')
                sys.exit(1)
        else:
            raise('There is no file here')
            sys.exit(1)
