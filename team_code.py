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
import tensorflow as tf
from scipy import stats as st
from sklearn.model_selection import KFold
import data_generator as dg
import graphing as plotter
from custom_train import *
# for reproducability
seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)

# only use cpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# parameters that can be modified
threshold = 48
################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

def model_lstm(timesteps, features_shape, num_classes):
    # delta model
    # timesteps x features
    delta_inputs = tf.keras.layers.Input(
        shape=(timesteps, features_shape[0])
    )
    delta_l1 = tf.keras.layers.Masking(mask_value=0.)(delta_inputs)
    delta_l2 = tf.keras.layers.LSTM(128)(delta_l1)
    delta_l3 = tf.keras.layers.Dense(32, activation='relu')(delta_l2)

    # theta model
    theta_inputs = tf.keras.layers.Input(
        shape=(timesteps, features_shape[1])
    )
    theta_l1 = tf.keras.layers.Masking(mask_value=0.)(theta_inputs)
    theta_l2 = tf.keras.layers.LSTM(128)(theta_l1)
    theta_l3 = tf.keras.layers.Dense(16, activation='relu')(theta_l2)

    # alpha model
    alpha_inputs = tf.keras.layers.Input(
        shape=(timesteps, features_shape[2])
    )
    alpha_l1 = tf.keras.layers.Masking(mask_value=0.)(alpha_inputs)
    alpha_l2 = tf.keras.layers.LSTM(128)(alpha_l1)
    alpha_l3 = tf.keras.layers.Dense(16, activation='relu')(alpha_l2)

    # beta model
    beta_inputs = tf.keras.layers.Input(
        shape=(timesteps, features_shape[3])
    )
    beta_l1 = tf.keras.layers.Masking(mask_value=0.)(beta_inputs)
    beta_l2 = tf.keras.layers.LSTM(128)(beta_l1)
    beta_l3 = tf.keras.layers.Dense(32, activation='relu')(beta_l2)
    
    # Merge all the models
    concatenated_layers = tf.keras.layers.concatenate([delta_l3, theta_l3, alpha_l3, beta_l3])

    concatenated_l4 = tf.keras.layers.Dense(32, activation='relu')(concatenated_layers)
    output_layer = tf.keras.layers.Dense(num_classes)(concatenated_l4)

    merged_model = tf.keras.models.Model(inputs=[(delta_inputs, theta_inputs, alpha_inputs, beta_inputs)], outputs=[output_layer])
    merged_model.summary()
    return merged_model


# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    # Create a folder for the graph if it does not already exist.
    graph_folder = os.path.join(model_folder, "graph")
    os.makedirs(graph_folder, exist_ok=True)
    

    # For submission
    if not os.path.exists(os.path.join(data_folder, 'validation')) or not os.path.exists(os.path.join(data_folder, 'training')):
        validation = False
        training_folder = data_folder
    else:
        validation = True
        validation_folder = os.path.join(data_folder, 'validation')
        training_folder = os.path.join(data_folder, 'training')
    
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids_train = find_data_folders(training_folder)

    if (validation):
        patient_ids_val = find_data_folders(validation_folder)
    num_patients_train = len(patient_ids_train)

    if num_patients_train==0:
        raise FileNotFoundError('No training data was provided.')

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    # PARAMETERS
    # time smaple x channel
    # hardcoding (but this is for delta psd iirc)
    timesteps = 72
    features = (342, 180, 180, 828)
    # dimension = (72, 342)
    num_classes = 2
    
    # Training parameters
    batch_size = 16
    epochs = 10

    # Create Data Generator
    if verbose >= 1:
        print('Creating data generator...')

    training_generator = dg.DataGenerator(patient_ids_train, training_folder, batch_size=batch_size, threshold=threshold)
    if validation:
        validation_generator = dg.DataGenerator(patient_ids_val, validation_folder, batch_size=batch_size, threshold=threshold)

    # Create Model
    model_outcome = model_lstm(timesteps, features, num_classes)

    # Train Model
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')

    # output of a custom fit is dictionary (for now)
    if validation:
        history_outcome = custom_fit(model_outcome, epochs, training_generator, validation_generator)
    else:
        history_outcome = custom_fit(model_outcome, epochs, training_generator)

    # Plot graph
    plotter.plot_loss_curve_dict(history_outcome, graph_folder)
    plotter.plot_accuracy_curve_dict(history_outcome, graph_folder)

    # Save model
    save_challenge_model_lstm(model_folder, model_outcome, "model_outcome")
    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    # filename = os.path.join(model_folder, 'models.sav')
    foldername_outcome = os.path.join(model_folder, 'model_outcome')
    # foldername_cpc = os.path.join(model_folder, 'model_cpc')
    
    # cpc_model = tf.keras.models.load_model(foldername_cpc)
    # cpc_model.summary()
    outcome_model = tf.keras.models.load_model(foldername_outcome)
    outcome_model.summary()
    # return joblib.load(filename), lstm_outcome, lstm_cpc 
    # return lstm_outcome, lstm_cpc, joblib.load(filename)
    return outcome_model

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    # lstm_model_outcome = models[0]
    # lstm_model_cpc = models[1]
    # random_tree_model = models[2]
    # imputer = random_tree_model['imputer']
    # outcome_model = random_tree_model['outcome_model']
    # cpc_model = random_tree_model['cpc_model']

    outcome_model = models

    # probability_model = tf.keras.Sequential([
    #     outcome_model,
    #     tf.keras.layers.Softmax()
    # ])
    # Load data.
    softmax_layer = tf.keras.layers.Softmax()(outcome_model.output)
    probability_model = tf.keras.models.Model(inputs=outcome_model.input, outputs=softmax_layer)
    probability_model.summary()
    patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)
    # Extract features.
    # patient_features, available_signal_data, delta_psd_data, theta_psd_data, alpha_psd_data, beta_psd_data = get_features(patient_metadata, recording_metadata, recording_data)
    # patient_features = patient_features.reshape(1, -1)
    patient_features, available_signal_data, delta_psd_data, theta_psd_data, alpha_psd_data, beta_psd_data = get_features(patient_metadata, recording_metadata, recording_data, threshold)
    print(np.shape(delta_psd_data))
    delta_psd_data = _arr_transformations_model(delta_psd_data)
    theta_psd_data = _arr_transformations_model(theta_psd_data)
    alpha_psd_data = _arr_transformations_model(alpha_psd_data)
    beta_psd_data = _arr_transformations_model(beta_psd_data)

    outcome_pred = probability_model.predict((delta_psd_data, theta_psd_data, alpha_psd_data, beta_psd_data))
    print("patient id: ", patient_id)
    print("outcome softmax: ", outcome_pred)
    print(np.shape(outcome_pred))
    # summed_outcome_pred = np.sum(outcome_pred, axis=0)
    # avg_outcome_pred = summed_outcome_pred / np.shape(outcome_pred)[0]
    # avg_outcome_pred = np.array([avg_outcome_pred])
    # 0 and 1 index is cpc 1 and 2, 2 3 4 index is cpc 3, 4, and 5
    # print(avg_outcome_pred)
    outcome_probability = outcome_pred[0][1]
    print("outcome_proba: ", outcome_probability)
    outcome_pred = np.argmax(outcome_pred, axis=1).astype(np.int64)
    print("outcome predict: ", outcome_pred)
    if (outcome_pred[0] < 1):
        # good
        cpc = 1
    else:
        # poor
        cpc = 5
    print("cpc:", cpc)
    # Impute missing data.
    # patient_features = imputer.transform(patient_features)

    # Apply models to features.
    # outcome = outcome_model.predict(patient_features)[0]
    # outcome_probability = outcome_model.predict_proba(patient_features)[0, 1]
    # cpc = cpc_model.predict(patient_features)[0]

    # outcome = np.round(lstm_model_outcome.predict(available_signal_data))
    # if not np.isnan(available_signal_data[0][0][0]):
    #     outcome_probability = lstm_model_outcome.predict(np.reshape(available_signal_data, (-1, 18, 30000)))
    #     outcome_probability = np.argmax(outcome_probability, axis=1).astype(np.int64)
    #     outcome_probability = np.average(outcome_probability)
    #     outcome = np.round(outcome_probability)
    # else:
    #     # use random forest model if there are no eeg data
    #     outcome = outcome_model.predict(patient_features)[0]
    #     outcome_probability = outcome_model.predict_proba(patient_features)[0, 1]
        # outcome_probability = np.random.random()
        # outcome = np.round(outcome_probability)
    # outcome_probability = st.mode(outcome, keepdims=False)
    # convert mode object to integer
    # outcome_probability = int(outcome_probability[0][0])
    # if not np.isnan(available_signal_data[0][0][0]):
    #     cpc = lstm_model_cpc.predict(np.reshape(available_signal_data, (-1, 18, 30000)))
    #     # reconvert to 1 - 5 cpc
    #     cpc = np.argmax(cpc, axis=1).astype(np.int64) + 1
    #     cpc = np.average(cpc)
    # else:
    #     # use random forest model if there is no eeg data
    #     cpc = cpc_model.predict(patient_features)[0]
    
    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    return outcome_pred, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################
def _arr_transformations_model(data_arr):
    """Convert from (X, Y, Z) into (X, Y * Z) into (72, Y * Z) into (1, 72, Y * Z) given X <= 72 to feed to the model
    """
    data_shape = np.shape(data_arr)
    data_arr = np.reshape(data_arr, (data_shape[0], data_shape[1] * data_shape[2]))
    data_arr = _pad_timeseries_arr(data_arr)
    data_shape = np.shape(data_arr)
    data_arr = np.reshape(data_arr, (1, *data_shape))
    data_arr = np.asarray(data_arr).astype(np.float32)
    return data_arr

def _pad_timeseries_arr(data_arr, desired_time_pad=72):
    current_shape = np.shape(data_arr)
    rows_to_pad = desired_time_pad - current_shape[0]
    padded_arr = np.pad(data_arr, ((0, rows_to_pad), (0, 0)), mode='constant')
    return padded_arr

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)

def save_challenge_model_lstm(model_folder, model, folder_name):
    os.makedirs(os.path.join(model_folder, folder_name), exist_ok=True)
    model.save(os.path.join(model_folder, folder_name))

# Extract features from the data.
def get_features_2(patient_metadata, recording_metadata, recording_data, threshold):
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

def create_model(
    input_channels:int,
    time_points:int,
    output_channels:int,
    n_estimators:int=8,
    dw_filters:int=8,
    cn_filters:int=16,
    sp_filters:int=32
):  
    """ EEGNet + RNN Model.
    input_channels: number of data's channel
    time_points: cumulated data's total time point
    output_channels: number of output node
    n_estimators: number of RNN cells
    dw_filters: number of filters for DepthwiseConv2D
    cn_filters: number of filters for Conv2D (1st filter from the paper EEGNet)
    sp_filters: number of filters for SeparableConv2D (2nd filter from the paper EEGNet)7\
    
    return  model: tf.keras.Model
    """
    # Define input layer
    inputs = tf.keras.layers.Input(
        shape=(
            input_channels, 
            time_points,
        )
    )
    cn = tf.keras.layers.Reshape((input_channels, time_points, 1))(inputs)
    cn = tf.keras.layers.Conv2D(filters=cn_filters, 
            kernel_size=(1, 64), 
            padding='same', 
            activation='linear',
            use_bias=False,
            )(cn)
    cn = tf.keras.layers.BatchNormalization()(cn)

    # Depthwise convolution layer
    dw = tf.keras.layers.DepthwiseConv2D(kernel_size=(input_channels, 1),
                        padding='valid',
                        depth_multiplier=dw_filters,
                        depthwise_constraint=tf.keras.constraints.max_norm(1.),
                        activation='linear',
                        use_bias=False,
                        )(cn)
    dw = tf.keras.layers.BatchNormalization()(dw)
    dw = tf.keras.layers.Activation('elu')(dw)
    dw = tf.keras.layers.AveragePooling2D(pool_size=(1, 4),
                        padding='valid',
                        )(dw)
    
    # Separable convolution layer
    sp = tf.keras.layers.SeparableConv2D(filters=sp_filters,
                        kernel_size=(1, 8),
                        padding='same',
                        activation='linear',
                        use_bias=False,
                        )(dw)
    sp = tf.keras.layers.BatchNormalization()(sp)
    sp = tf.keras.layers.Activation('elu')(sp)

    # # RNN layer
    # shape = tuple([x for x in sp.shape.as_list() if x != 1 and x is not None])
    # sp = tf.keras.layers.Reshape(shape)(sp)
    # sp = tf.keras.layers.GRU(n_estimators, return_sequences=True)(sp)
    # sp = tf.keras.layers.Dropout(0.5)(sp)

    # Flatten output
    sp = tf.keras.layers.Flatten()(sp)

    # Output layer
    outputs = tf.keras.layers.Dense(output_channels,
                    kernel_constraint=tf.keras.constraints.max_norm(0.25),
                    )(sp)
    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

# Extract features from the data.
def get_features(patient_metadata, recording_metadata, recording_data, threshold):
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
        available_signal_data = np.empty((1, 30000, 18))
        available_signal_data.fill(np.NaN)
    
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
    # return patient_features, available_signal_data, delta_psd_data, theta_psd_data, alpha_psd_data, beta_psd_data

