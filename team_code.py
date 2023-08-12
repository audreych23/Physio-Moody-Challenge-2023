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
import tensorflow as tf
from scipy import stats as st
from sklearn.model_selection import KFold
import data_generator as dg
import graphing as plotter
from custom_train import *
from models import *
from save_model import * 
import train_and_evaluate_model
# for reproducability
seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)

# only use cpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# parameters that can be modified

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Disable This if validation is not used
    validation = True
    # Disable if k_fold is not used
    use_k_fold_cross_validation = True
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    # Create a folder for the graph if it does not already exist.
    graph_folder = os.path.join(model_folder, "graph")
    os.makedirs(graph_folder, exist_ok=True)
    
    # PARAMETERS
    data_split_validation = 20
    # time smaple x channel
    # hardcoding (but this is for delta psd iirc)
    timesteps = 72
    features = (342, 180, 180, 828)
    # dimension = (72, 342)
    num_classes = 2
    
    # Training parameters
    batch_size = 16
    epochs = 2
    threshold = 48
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    print(patient_ids)
    
    total_num_patients_train = len(patient_ids)

    if total_num_patients_train==0:
        raise FileNotFoundError('No training data was provided.')

    # First is for evaluation how good the model is
    if use_k_fold_cross_validation:
        train_and_evaluate_model.k_fold_cross_validation(data_folder, model_folder, graph_folder, verbose, 
                                                   patient_ids, timesteps, features, num_classes, batch_size, epochs, k = 5)
    else:
        train_and_evaluate_model.train_and_evaluate_model(data_folder, model_folder, graph_folder, verbose, 
                                                   patient_ids, timesteps, features, num_classes, batch_size, epochs, validation, data_split_validation)

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
    test_data_generator = dg.DataGenerator(patient_ids, data_folder, batch_size=1, to_fit=False, shuffle=False)
        
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
