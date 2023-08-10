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
from models import *
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

    training_generator = dg.DataGenerator(patient_ids_train, data_folder, batch_size=batch_size, threshold=threshold)
    if validation:
        validation_generator = dg.DataGenerator(patient_ids_val, data_folder, batch_size=batch_size, threshold=threshold)

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

    softmax_layer = tf.keras.layers.Softmax()(outcome_model.output)
    probability_model = tf.keras.models.Model(inputs=outcome_model.input, outputs=softmax_layer)
    probability_model.summary()
    
    # batch size has to be 1 because only one data per run
    patient_ids = list()
    patient_ids.append(patient_id)
    # Load data.
    test_data_generator = dg.DataGenerator(patient_ids, data_folder, batch_size=1, threshold=threshold, to_fit=False, shuffle=False)
        
    outcome_pred = probability_model.predict_generator(test_data_generator)

    print("patient id: ", patient_id)
    print("outcome softmax: ", outcome_pred)
    print(np.shape(outcome_pred))

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

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    return outcome_pred, outcome_probability, cpc

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
    os.makedirs(os.path.join(model_folder, folder_name), exist_ok=True)
    model.save(os.path.join(model_folder, folder_name))
