import tensorflow as tf
import numpy as np
import helper_code as hp

def preload_y_data(patient_ids, data_path):
    y_data = list()
    # Generate data
    for i, patient_id in enumerate(patient_ids):
        # Store sample
        patient_metadata, recording_metadata, recording_data = hp.load_challenge_data(data_path, patient_id)

        outcome_data = hp.get_outcome(patient_metadata)
        y_data.append(outcome_data)

    return y_data
