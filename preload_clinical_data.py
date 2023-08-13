import tensorflow as tf
import numpy as np
import helper_code as hp

def preload_clinical_data(patient_ids, data_folder):
    list_patient_features = list()
    for patient_id in patient_ids:
        patient_metadata, _, _ = hp.load_challenge_data(data_folder, patient_id)
        patient_features = get_clinical_features(patient_metadata)
        list_patient_features.append(patient_features)
    
    list_patient_features = np.array(list_patient_features)
    list_patient_features = np.vstack(list_patient_features)

    print(np.shape(list_patient_features))
    return

def get_clinical_features(patient_metadata):
    # Extract features from the patient metadata.
    age = hp.get_age(patient_metadata)
    sex = hp.get_sex(patient_metadata)
    rosc = hp.get_rosc(patient_metadata)
    ohca = hp.get_ohca(patient_metadata)
    vfib = hp.get_vfib(patient_metadata)
    ttm = hp.get_ttm(patient_metadata)

    # Use one-hot encoding for sex; add more variables
    sex_features = np.zeros(2, dtype=int)
    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    # Combine the patient features.
    patient_features = np.array([age, female, male, other, rosc, ohca, vfib, ttm])