#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
import mne
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

def prepare_label(model_type, patient_metadata, available_signal_data):
    # model type is an integer from 1, 2, 3
    outcomes = list()
    cpcs = list()
    if (model_type == 1):
        current_outcome = get_outcome(patient_metadata)
        current_cpc = get_cpc(patient_metadata)
        for i in range(available_signal_data.shape[0]):
            outcomes.append(current_outcome)
            cpcs.append(current_cpc)
    else:
        raise Exception("this model_type have not been implemented")
    
    return outcomes, cpcs

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    print_flag = 1
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    patients_features = list()
    available_signal_datas = list()
    delta_psd_datas = list() 
    theta_psd_datas = list() 
    alpha_psd_datas = list() 
    beta_psd_datas = list()

    outcomes = list()
    cpcs = list()

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        # Load data.
        patient_id = patient_ids[i]
        patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

        # Extract features.
        # current_features = get_features(patient_metadata, recording_metadata, recording_data)
        # current_features = get_features_test(patient_metadata, recording_metadata, recording_data)
        # features.append(current_features)
        patient_features, available_signal_data, delta_psd_data, theta_psd_data, alpha_psd_data, beta_psd_data = get_features_test(patient_metadata, recording_metadata, recording_data)

        patients_features.append(patient_features)
        available_signal_datas.append(available_signal_data)
        delta_psd_datas.append(delta_psd_data)
        theta_psd_datas.append(theta_psd_data)
        alpha_psd_datas.append(alpha_psd_data)
        beta_psd_datas.append(beta_psd_data)

        if print_flag == 1:
            print("patient features shape: ", patient_features.shape)
            print("available signal shape: ", available_signal_data.shape)
            print("delta psd shape: ", delta_psd_data.shape)
            print("theta psd shape: ", theta_psd_data.shape)
            print("alpha psd shape: ", alpha_psd_data.shape)
            print("beta psd shape: ", beta_psd_data.shape)

        # Extract labels.
        try:
            outcomes, cpcs = prepare_label(model_type=1, patient_metadata=patient_metadata, available_signal_data=available_signal_data)
        except:
            print("model_type has not been implemented, exiting.....")
            exit(1)

    patients_features = np.vstack(patients_features)
    available_signal_datas = np.vstack(available_signal_datas)
    delta_psd_datas = np.vstack(delta_psd_datas)
    theta_psd_datas = np.vstack(theta_psd_datas)
    alpha_psd_datas = np.vstack(alpha_psd_datas)
    beta_psd_datas = np.vstack(beta_psd_datas)

    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)
    
    if (print_flag == 1):
        # sanity check
        print("patients features shape", patients_features.shape)
        print("available signal datas shape", available_signal_datas.shape)
        print("delta psd datas shape", delta_psd_datas.shape)
        print("theta psd datas shape", theta_psd_datas.shape)
        print("alpha psd datas shape", alpha_psd_datas.shape)
        print("beta psd datas", beta_psd_datas.shape)
        print("outcomes shape", outcomes.shape)
        print("cpcs shape", cpcs.shape)


    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')

    # Define parameters for random forest classifier and regressor.
    n_estimators   = 123  # Number of trees in the forest.
    max_leaf_nodes = 456  # Maximum number of leaf nodes in each tree.
    random_state   = 789  # Random state; set for reproducibility.

    # Impute any missing features; use the mean value by default.
    imputer = SimpleImputer().fit(features)

    # Train the models.
    features = imputer.transform(features)
    outcome_model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, outcomes.ravel())
    cpc_model = RandomForestRegressor(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, cpcs.ravel())

    # Save the models.
    save_challenge_model(model_folder, imputer, outcome_model, cpc_model)

    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename)

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    imputer = models['imputer']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']

    # Load data.
    patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

    # Extract features.
    features = get_features(patient_metadata, recording_metadata, recording_data)
    features = features.reshape(1, -1)

    # Impute missing data.
    features = imputer.transform(features)

    # Apply models to features.
    outcome = outcome_model.predict(features)[0]
    outcome_probability = outcome_model.predict_proba(features)[0, 1]
    cpc = cpc_model.predict(features)[0]

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    return outcome, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)

# Extract features from the data.
def get_features(patient_metadata, recording_metadata, recording_data):
    # TODO : DELETE THIS
    verbose = 1
    # Extract features from the patient metadata.
    age = get_age(patient_metadata)
    sex = get_sex(patient_metadata)
    rosc = get_rosc(patient_metadata)
    ohca = get_ohca(patient_metadata)
    vfib = get_vfib(patient_metadata)
    ttm = get_ttm(patient_metadata)

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

    # Extract features from the recording data and metadata.
    channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
                'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    num_channels = len(channels)
    num_recordings = len(recording_data)

    # Compute mean and standard deviation for each channel for each recording.
    available_signal_data = list()
    for i in range(num_recordings):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.
            available_signal_data.append(signal_data)
    
    if (verbose >= 1):
        print("available signal_data[0] shape :", available_signal_data[0].shape)
        print("total amount hours", len(available_signal_data))
        # print(signal_data[0])
        # print("signal_data shape: ", signal_data.shape)

    if len(available_signal_data) > 0:
        available_signal_data = np.hstack(available_signal_data)
        signal_mean = np.nanmean(available_signal_data, axis=1)
        signal_std  = np.nanstd(available_signal_data, axis=1)
    else:
        signal_mean = float('nan') * np.ones(num_channels)
        signal_std  = float('nan') * np.ones(num_channels)

    # Compute the power spectral density for the delta, theta, alpha, and beta frequency bands for each channel of the most
    # recent recording.
    index = None
    for i in reversed(range(num_recordings)):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            index = i
            break

    if index is not None:
        signal_data, sampling_frequency, signal_channels = recording_data[index]
        signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.

        delta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False)
        theta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
        beta_psd,  _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)

        if (verbose >= 1):
            print("delta_psd shape :", delta_psd.shape)
            print("theta_psd shape :", theta_psd.shape)
            print("theta_psd shape :", alpha_psd.shape)
            print("beta_psd shape :", beta_psd.shape)

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean  = np.nanmean(beta_psd,  axis=1)

        quality_score = get_quality_scores(recording_metadata)[index]
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)
        quality_score = float('nan')

    recording_features = np.hstack((signal_mean, signal_std, delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean, quality_score))

    if (verbose >= 1):
        print("recording features shape:", recording_features.shape)
    # Combine the features from the patient metadata and the recording data and metadata.
    features = np.hstack((patient_features, recording_features))

    if (verbose >= 1):
        print("features shape:", features.shape)

    return features

# Extract features from the data.
def get_features_test(patient_metadata, recording_metadata, recording_data):
    """ Get the Timestamps Data.
        @params
            * patient_metadata (list):
                Data of Patient.
            * recording_metadata (list):
                Data of Recording.
            * recording_data (list):
                Recording Data.
            * option (int):
                * 1: Recording timestamp with PSDs.
                * 2: Recording timestamp with PSDs and appended Quality.
        @return
            Features, format according to the selected option.
    """
    verbose = 1
    # PARAM
    add_quality = False
    return_by_hours = True

    # Extract features from the patient metadata.
    age = get_age(patient_metadata)
    sex = get_sex(patient_metadata)
    rosc = get_rosc(patient_metadata)
    ohca = get_ohca(patient_metadata)
    vfib = get_vfib(patient_metadata)
    ttm = get_ttm(patient_metadata)
    quality_scores = get_quality_scores(recording_metadata)

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

    # Extract features from the recording data and metadata.
    channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
                'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    num_channels = len(channels)
    num_recordings = len(recording_data)

    # 1. Compute mean and standard deviation for each channel for each recording.
    # 2. Compute the power spectral density for the delta, theta, alpha, and beta frequency bands 
    #    for each channel from all the recording.
    available_signal_data = list()
    delta_psd_data = list()
    theta_psd_data = list()
    alpha_psd_data = list()
    beta_psd_data  = list()

    for i in range(num_recordings):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.
        
            delta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False)
            theta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
            alpha_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
            beta_psd,  _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)
            quality_score = list()
            quality_score.append(quality_scores[i])

            if add_quality == True: # num_channels (+1)
                signal_data = np.append(signal_data, [quality_score * signal_data.shape[1]], axis=0)
                delta_psd   = np.append(delta_psd, [quality_score * delta_psd.shape[1]], axis=0)
                theta_psd   = np.append(theta_psd, [quality_score * theta_psd.shape[1]], axis=0)
                alpha_psd   = np.append(alpha_psd, [quality_score * alpha_psd.shape[1]], axis=0)
                beta_psd    = np.append(beta_psd, [quality_score * beta_psd.shape[1]], axis=0)                

            # DEBUG
            # print("shapes:", signal_data.shape, delta_psd.shape, theta_psd.shape, alpha_psd.shape, beta_psd.shape)
            
            available_signal_data.append(signal_data)
            delta_psd_data.append(delta_psd)
            theta_psd_data.append(theta_psd)
            alpha_psd_data.append(alpha_psd)
            beta_psd_data.append(beta_psd)
            

    if len(available_signal_data) > 0:

        if return_by_hours == True:
            stack_func = np.dstack            
            trans_axes = (2, 1, 0)
        else:
            stack_func = np.hstack
            trans_axes = (1, 0)
         
        available_signal_data = stack_func(available_signal_data)
        delta_psd_data = stack_func(delta_psd_data)
        theta_psd_data = stack_func(theta_psd_data)
        alpha_psd_data = stack_func(alpha_psd_data)
        beta_psd_data = stack_func(beta_psd_data)

        available_signal_data = np.transpose(available_signal_data, trans_axes)
        delta_psd_data = np.transpose(delta_psd_data, trans_axes)
        theta_psd_data = np.transpose(theta_psd_data, trans_axes)
        alpha_psd_data = np.transpose(alpha_psd_data, trans_axes)
        beta_psd_data = np.transpose(beta_psd_data, trans_axes)

    else:
        available_signal_data = delta_psd_data = theta_psd_data = alpha_psd_data = beta_psd_data = np.hstack(float('nan') * np.ones(num_channels))
    
    # DEBUG
    # print("available_signal_data:", available_signal_data.shape) 
    # print("delta_psd_data       :", delta_psd_data.shape)
    # print("theta_psd_data       :", theta_psd_data.shape)
    # print("alpha_psd_data       :", alpha_psd_data.shape)
    # print("beta_psd_data        :", beta_psd_data.shape)

    # if (verbose >= 1):
    #     print("recording features shape:", recording_features.shape)
    # # Combine the features from the patient metadata and the recording data and metadata.
    # features = np.hstack((patient_features, recording_features))

    
    # features = np.hstack((patient_features, available_signal_data, delta_psd_data, theta_psd_data, alpha_psd_data, beta_psd_data))
    # if (verbose >= 1):
    #     print("features shape:", features.shape)
    # Combine the features from the patient metadata and the recording data and metadata.
    return patient_features, available_signal_data, delta_psd_data, theta_psd_data, alpha_psd_data, beta_psd_data