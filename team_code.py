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
import h5py
import gc
# for reproducability
seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)

# uncomment this for gpu
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

# Create 2 logical gpus ***comment this before submission***
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   # Create 2 virtual GPUs with 1GB memory each
#     try:
#         tf.config.set_logical_device_configuration(
#             gpus[1],
#             [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
#             tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)

    
################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# call this if you want to check for cross validation
def k_fold_cross_validation(available_signal_datas, outcomes, cpcs, prefix_sum_index, num_patients, outcomes_random_forest):
    validation_accuracies_outcome = list()
    validation_losses_outcome = list()

    validation_accuracies_cpc = list()
    validation_losses_cpc = list()
    print_flag = 1
    kf = KFold(n_splits=5)
    # dummy variable to split the train and val_index
    for i, (train_index, val_index) in enumerate(kf.split(np.zeros((num_patients, 1)), outcomes_random_forest)):
        # Split the data into train and validation with k-fold cross validation
        if (print_flag == 1):
            print("train_index, val_index", train_index, val_index)
        # prepare x data for training and validation
        x_train = list()
        x_val = list()

        # prepare y data for training and validation
        y_train_outcome = list()
        y_val_outcome = list()
        y_train_cpc = list()
        y_val_cpc = list()
        # equivalent on using iloc
        for idx in val_index:
            start = prefix_sum_index[idx]
            end = prefix_sum_index[idx + 1]
            # use the matrix from [start: end] and concatenate with the next start, next end
            x_val.append(available_signal_datas[start:end])
            y_val_outcome.append(outcomes[start:end])
            y_val_cpc.append(cpcs[start:end])

        for idx in train_index:
            start = prefix_sum_index[idx]
            end = prefix_sum_index[idx + 1]
            # use the matrix from [start: end] and concatenate with the next start, next end
            x_train.append(available_signal_datas[start:end])
            y_train_outcome.append(outcomes[start:end])
            y_train_cpc.append(cpcs[start:end])
        
        # maybe this is why it broke
        x_val = np.vstack(x_val)
        x_train = np.vstack(x_train)
        y_train_outcome = np.vstack(y_train_outcome)
        y_val_outcome = np.vstack(y_val_outcome)
        y_train_cpc = np.vstack(y_train_cpc)
        y_val_cpc = np.vstack(y_val_cpc)

        if (print_flag == 1):
            # sanity check
            print("x_val", x_val.shape)
            print("x_train", x_train.shape)
            print("y_train_outcome", y_train_outcome.shape)
            print("y_val_outcome", y_val_outcome.shape)
            print("y_train_cpc", y_train_cpc.shape)
            print("y_val_cpc", y_val_cpc.shape)

        # Train the models.
        if print_flag >= 1:
            print('Training the Challenge models on the Challenge data...')

        # unusable right now don't call this function !
        # model_lstm_outcome
        model_lstm_outcome = create_model_lstm(x_train, 0)
        model_lstm_outcome.summary()
        # model_lstm_outcome = compile_model(x_train, y_train_outcome, x_val, y_val_outcome, model_lstm_outcome, 0)

        # evaluate the model
        results = model_lstm_outcome.evaluate(x=x_val, y=y_val_outcome)
        results = dict(zip(model_lstm_outcome.metrics_names, results))
        
        validation_accuracies_outcome.append(results['accuracy'])
        validation_losses_outcome.append(results['loss'])
        print(i, "- outcome validation accuracy, loss", results['accuracy'], results['loss'])

        # model_lstm_cpc
        model_lstm_cpc = create_model_lstm(x_train, 1)
        model_lstm_cpc.summary()
        # model_lstm_cpc = compile_model(x_train, y_train_cpc, x_val, y_val_cpc, model_lstm_cpc, 1)

        results = model_lstm_cpc.evaluate(x=x_val, y=y_val_cpc)
        results = dict(zip(model_lstm_cpc.metrics_names, results))

        validation_accuracies_cpc.append(results['accuracy'])
        validation_losses_cpc.append(results['loss'])

        tf.keras.backend.clear_session()
        if print_flag >= 1:
            print("Done", i, "training")

    for i, v in enumerate(validation_accuracies_outcome):
        print(i, '- (accuracy, losses) = ', validation_accuracies_outcome[i], validation_losses_outcome[i])

    avg_accuracy_outcome = average(validation_accuracies_outcome)
    avg_loss_outcome = average(validation_losses_outcome)
    print("avg accuracy:", avg_accuracy_outcome, ", avg loss:", avg_loss_outcome)

    for i, v in enumerate(validation_accuracies_cpc):
        print(i, '- (accuracy, losses) = ', validation_accuracies_cpc[i], validation_losses_cpc[i])

    avg_accuracy_outcome = average(validation_accuracies_outcome)
    avg_loss_outcome = average(validation_losses_outcome)
    print("avg accuracy:", avg_accuracy_outcome, ", avg loss:", avg_loss_outcome)

# return a model
def create_model_lstm(input_data, output_type):
    """
        param:
            input_data: a nd array with 3 dimensions (batch, timesteps, features)
            output_type: 0 or 1, where 0 is outcome and 1 is cpc
        returns
            model: tf.keras.models.Model type
    """
    inputs = tf.keras.layers.Input(
        shape=(input_data.shape[1], input_data.shape[2])
    )
    x = tf.keras.layers.LSTM(4)(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    if output_type == "outcome":
        outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    else:
        outputs = tf.keras.layers.Dense(5, activation='softmax')(x)

    return tf.keras.models.Model(inputs, outputs)

# def save_data():
#     # changable
#     data_path = "/mnt/usb/jupyter-audreych/saved_data"
#     if (not os.path.exists(data_path)):
#         os.makedirs(data_path)
    
#     h5f = h5py.file('available_signal_data', data=available_signal_data)
# modify hyper parameter here, and the amount of data to be processed can also be modifed (after the modulo in train_challenge_model function
def train_model(model, x_train, y_train, x_val=None, y_val=None, output_type="cpc", batch_size=32, epochs=5):
    """
        param:
            x_train: a nd array with 3 dimensions (batch, timesteps, features)
            y_train: a nd array with 2 dimensions (batch, 1)
            x_val: None if there is no validation data default None
            y_val: None if there is no validation data default None
            model : the model you want to train
            output_type: a string of "cpc" or "outcome"
        returns 
            the trained model
    """
    if output_type == "outcome":
        y_train = tf.keras.utils.to_categorical(y_train, 2)
        if (y_val != None):
            y_val = tf.keras.utils.to_categorical(y_val, 2)
    else:
        # convert to one-hot
        y_train = tf.keras.utils.to_categorical(y_train, 5)
        if (y_val != None):
            y_val = tf.keras.utils.to_categorical(y_val, 5)
    if (y_val != None):
        model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs)
    else:
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    
    return model
    

def compile_model(model, lr=0.00001, loss='categorical_crossentropy', metrics=['accuracy']):
    """
        param:
            model: the model wish to be compiled
            lr : learning_rate
            loss : the loss function for training model
            metrics : the metrics to evaluate the model
        returns
            the model compiled
    """
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=metrics)
    return model
    # model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10)
    

def prepare_label(model_type, patient_metadata, available_signal_data):
    # model type is an integer from 1, 2, 3
    outcomes = list()
    cpcs = list()
    if model_type == 1:
        current_outcome = get_outcome(patient_metadata)
        current_cpc = get_cpc(patient_metadata)
        for i in range(available_signal_data.shape[0]):
            outcomes.append(current_outcome)
            # force it to be 0 to 4 cpc then reconvert it later
            cpcs.append(current_cpc - 1)
    elif model_type == 2:
        current_outcome = get_outcome(patient_metadata)
        current_cpc = get_cpc(patient_metadata)
        outcomes.append(current_outcome)
        cpcs.append(current_cpc)
    else:
        raise Exception("this model_type have not been implemented")
    
    return outcomes, cpcs

# average of a list
def average(lst):
    return sum(lst)/len(lst)

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # for debugging purpose
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

    # list for x datas
    patients_features = list()
    available_signal_datas = list()
    delta_psd_datas = list() 
    theta_psd_datas = list() 
    alpha_psd_datas = list() 
    beta_psd_datas = list()

    # list for y datas : outcomes and cpcs
    outcomes = list()
    cpcs = list()
    outcomes_random_forest = list()
    cpcs_random_forest = list()

    # this is a list where the index of the list is equal to the current element on the list
    # say that on the first patient we have 50 hours of eeg recording, 2nd has 30 hours
    # prefix_sum_index[1] = 50, prefix_sum_index[2] = 50 + 30 = 80
    prefix_sum_index = list()
    prefix_sum_index.append(0)

    num_patients_stored = 0
    # model creation here : input dummy shape 
    # model_lstm_outcome = create_lstm_model(np.zeros((1, 18, 30000)), "outcome")
    # model_lstm_outcome.summary()
    # model_lstm_outcome = compile_model(model_lstm_outcome)
    
    # model_lstm_cpc = create_model_lstm(np.zeros((1, 18, 30000)), "cpc")
    # model_lstm_cpc.summary()
    # model_lstm_cpc = compile_model(model_lstm_cpc)
    model_lstm_outcome = create_model(18, 30000, 2)
    model_lstm_outcome = compile_model(model_lstm_outcome)
    model_lstm_cpc = create_model(18, 30000, 5)
    model_lstm_cpc = compile_model(model_lstm_cpc)

    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')
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
        patient_features, available_signal_data, delta_psd_data, theta_psd_data, alpha_psd_data, beta_psd_data = get_features(patient_metadata, recording_metadata, recording_data)
        patients_features.append(patient_features)
        if not np.isnan(available_signal_data[0][0][0]):
            # need to reshape??, this is used for k-cross validation
            # patient_features = patient_features.reshape(1, -1)
            # reshape for eeg net
            available_signal_datas.append(np.reshape(available_signal_data, (-1, 18, 30000)))
            num_patients_stored += 1
            # delta_psd_datas.append(delta_psd_data)
            # theta_psd_datas.append(theta_psd_data)
            # alpha_psd_datas.append(alpha_psd_data)
            # beta_psd_datas.append(beta_psd_data)
            # ignore this for now
            # prefix_sum_index.append(prefix_sum_index[i] + available_signal_data.shape[0])

        # if print_flag == 1:
        #     # sanity check
        #     print("patient features shape: ", patient_features.shape)
        #     print("available signal shape: ", available_signal_data.shape)
        #     print("delta psd shape: ", delta_psd_data.shape)
        #     print("theta psd shape: ", theta_psd_data.shape)
        #     print("alpha psd shape: ", alpha_psd_data.shape)
        #     print("beta psd shape: ", beta_psd_data.shape)
        #     print('prefix_sum_index', i + 1, prefix_sum_index[i + 1])

        # Extract labels.
        try:
            if not np.isnan(available_signal_data[0][0][0]):
                outcome, cpc = prepare_label(model_type=1, patient_metadata=patient_metadata, available_signal_data=available_signal_data)
                outcomes.extend(outcome)
                cpcs.extend(cpc)
            
            outcome_random_forest, cpc_random_forest = prepare_label(model_type=2, patient_metadata=patient_metadata, available_signal_data=available_signal_data) 
            outcomes_random_forest.extend(outcome_random_forest)
            cpcs_random_forest.extend(cpc_random_forest)

        except:
            print("model_type has not been implemented, exiting.....")
            exit(1)
        
        if (num_patients_stored != 0 and num_patients_stored % 5 == 0):
            available_signal_datas = np.vstack(available_signal_datas)
            outcomes = np.vstack(outcomes)
            cpcs = np.vstack(cpcs)

            # train the model
            model_lstm_outcome = train_model(model_lstm_outcome, available_signal_datas, outcomes, None, None, "outcome")
            model_lstm_cpc = train_model(model_lstm_cpc, available_signal_datas, cpcs, None, None, "cpc")
            
            # clear the array
            available_signal_datas = list()
            outcomes = list()
            cpcs = list()
            gc.collect()

    # delta_psd_datas = np.vstack(delta_psd_datas)
    # theta_psd_datas = np.vstack(theta_psd_datas)
    # alpha_psd_datas = np.vstack(alpha_psd_datas)
    # beta_psd_datas = np.vstack(beta_psd_datas)
    if (len(available_signal_datas) != 0):
        available_signal_datas = np.vstack(available_signal_datas)
        outcomes = np.vstack(outcomes)
        cpcs = np.vstack(cpcs)
    patients_features = np.vstack(patients_features)
    
    outcomes_random_forest = np.vstack(outcomes_random_forest)
    cpcs_random_forest = np.vstack(cpcs_random_forest)

    # if (print_flag == 1):
    #     # sanity check
    #     print("patients features shape", patients_features.shape)
    #     if (len(available_signal_datas) != 0):
    #     print("available signal datas shape", available_signal_datas.shape)
    #     # print("delta psd datas shape", delta_psd_datas.shape)
    #     # print("theta psd datas shape", theta_psd_datas.shape)
    #     # print("alpha psd datas shape", alpha_psd_datas.shape)
    #     # print("beta psd datas", beta_psd_datas.shape)
    #     print("outcomes shape", outcomes.shape)
    #     print("cpcs shape", cpcs.shape)
    #     print("outcomes_random_forest", outcomes_random_forest.shape)
    #     print("cpcs_random_forest", cpcs_random_forest.shape)

    # train with whole data
    # for reproducible purpose
    # with gpu
    # tf.debugging.set_log_device_placement(True)
    # gpus = tf.config.list_logical_devices('GPU')
    # strategy = tf.distribute.MirroredStrategy(gpus)
    # with strategy.scope():
    #     model_lstm_outcome = create_model_lstm(available_signal_datas, 0)
    #     model_lstm_outcome.summary()
    #     model_lstm_outcome = compile_train_model(available_signal_datas, outcomes, None, None, model_lstm_outcome, 0)
    #     save_challenge_model_lstm(model_folder, model_lstm_outcome, "model_outcome")
        
    #     model_lstm_cpc = create_model_lstm(available_signal_datas, 1)
    #     model_lstm_cpc.summary()
    #     model_lstm_cpc = compile_train_model(available_signal_datas, cpcs, None, None, model_lstm_cpc, 1)
    #     save_challenge_model_lstm(model_folder, model_lstm_cpc, "model_cpc")

    # available signal datas has to be reshape for this model 
    # available_signal_datas = available_signal_datas.reshape((available_signal_datas.shape[0], available_signal_datas.shape[2], available_signal_data.shape[1]))
    # model_outcome = create_model(available_signal_datas.shape[1], available_signal_datas.shape[2], 2)
    # model_outcome.summary()
    # model_outcome = compile_train_model(available_signal_datas, outcomes, None, None, model_outcome, "outcome")
    # save_challenge_model_lstm(model_folder, model_outcome, "model_outcome")

    # model_cpc = create_model(available_signal_datas.shape[1], available_signal_datas.shape[2], 5)
    # model_cpc.summary()
    # model_cpc = compile_train_model(available_signal_datas, cpcs, None, None, model_cpc, "cpc")
    # save_challenge_model_lstm(model_folder, model_cpc, "model_cpc")
    # without gpu
    # model_lstm_outcome = create_model_lstm(available_signal_datas, "outcome")
    # model_lstm_outcome.summary()
    # model_lstm_outcome = compile_train_model(available_signal_datas, outcomes, None, None, model_lstm_outcome, "outcome")
    if (available_signal_datas.size != 0):
        model_lstm_outcome = train_model(model_lstm_outcome, available_signal_datas, outcomes, None, None, "outcome")
    save_challenge_model_lstm(model_folder, model_lstm_outcome, "model_outcome")
    
    # model_lstm_cpc = create_model_lstm(available_signal_datas, "cpc")
    # model_lstm_cpc.summary()
    # model_lstm_cpc = compile_train_model(available_signal_datas, cpcs, None, None, model_lstm_cpc, "cpc")
    if (available_signal_datas.size != 0):
        model_lstm_cpc = train_model(model_lstm_cpc, available_signal_datas, cpcs, None, None, "cpc")
    save_challenge_model_lstm(model_folder, model_lstm_cpc, "model_cpc")

    # Define parameters for random forest classifier and regressor.
    n_estimators   = 128  # Number of trees in the forest.
    max_leaf_nodes = 512  # Maximum number of leaf nodes in each tree.
    random_state   = 1  # Random state; set for reproducibility.

    # Impute any missing features; use the mean value by default.
    imputer = SimpleImputer().fit(patients_features)

    # # Train the models.
    patients_features = imputer.transform(patients_features)
    outcome_model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(patients_features, outcomes_random_forest.ravel())
    cpc_model = RandomForestRegressor(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(patients_features, cpcs_random_forest.ravel())
    # train the model again with the whole dataset
    # Save the models.
    save_challenge_model(model_folder, imputer, outcome_model, cpc_model)

    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    foldername_lstm_outcome = os.path.join(model_folder, 'model_outcome')
    foldername_lstm_cpc = os.path.join(model_folder, 'model_cpc')
    lstm_outcome = tf.keras.models.load_model(foldername_lstm_outcome)
    lstm_outcome.summary()
    lstm_cpc = tf.keras.models.load_model(foldername_lstm_cpc)
    lstm_cpc.summary()
    # return joblib.load(filename), lstm_outcome, lstm_cpc 
    return lstm_outcome, lstm_cpc, joblib.load(filename)

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    lstm_model_outcome = models[0]
    lstm_model_cpc = models[1]
    random_tree_model = models[2]
    imputer = random_tree_model['imputer']
    outcome_model = random_tree_model['outcome_model']
    cpc_model = random_tree_model['cpc_model']

    # Load data.
    patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

    # Extract features.
    patient_features, available_signal_data, delta_psd_data, theta_psd_data, alpha_psd_data, beta_psd_data = get_features(patient_metadata, recording_metadata, recording_data)
    patient_features = patient_features.reshape(1, -1)

    # Impute missing data.
    patient_features = imputer.transform(patient_features)

    # Apply models to features.
    # outcome = outcome_model.predict(patient_features)[0]
    # outcome_probability = outcome_model.predict_proba(patient_features)[0, 1]
    # cpc = cpc_model.predict(patient_features)[0]

    # outcome = np.round(lstm_model_outcome.predict(available_signal_data))
    if not np.isnan(available_signal_data[0][0][0]):
        outcome_probability = lstm_model_outcome.predict(np.reshape(available_signal_data, (-1, 18, 30000)))
        outcome_probability = np.argmax(outcome_probability, axis=1).astype(np.int64)
        outcome_probability = np.average(outcome_probability)
        outcome = np.round(outcome_probability)
    else:
        # use random forest model if there are no eeg data
        outcome = outcome_model.predict(patient_features)[0]
        outcome_probability = outcome_model.predict_proba(patient_features)[0, 1]
        # outcome_probability = np.random.random()
        # outcome = np.round(outcome_probability)
    # outcome_probability = st.mode(outcome, keepdims=False)
    # convert mode object to integer
    # outcome_probability = int(outcome_probability[0][0])
    if not np.isnan(available_signal_data[0][0][0]):
        cpc = lstm_model_cpc.predict(np.reshape(available_signal_data, (-1, 18, 30000)))
        # reconvert to 1 - 5 cpc
        cpc = np.argmax(cpc, axis=1).astype(np.int64) + 1
        cpc = np.average(cpc)
    else:
        # use random forest model if there is no eeg data
        cpc = cpc_model.predict(patient_features)[0]
    
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

def save_challenge_model_lstm(model_folder, model, folder_name):
    model.save(os.path.join(model_folder, folder_name))

# Extract features from the data.
def get_features_2(patient_metadata, recording_metadata, recording_data):
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

    # RNN layer
    shape = tuple([x for x in sp.shape.as_list() if x != 1 and x is not None])
    sp = tf.keras.layers.Reshape(shape)(sp)
    sp = tf.keras.layers.GRU(n_estimators, return_sequences=True)(sp)
    sp = tf.keras.layers.Dropout(0.5)(sp)

    # Flatten output
    sp = tf.keras.layers.Flatten()(sp)

    # Output layer
    outputs = tf.keras.layers.Dense(output_channels,
                    activation='softmax',
                    kernel_constraint=tf.keras.constraints.max_norm(0.25),
                    )(sp)
    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

# Extract features from the data.
def get_features(patient_metadata, recording_metadata, recording_data):
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

