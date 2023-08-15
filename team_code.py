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
import tensorflow as tf
import model_training.train_and_evaluate_model as train_and_evaluate_model
import random
# for reproducability
seed = 1
random.seed(seed)
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
    features = (1530, 8)
    # dimension = (72, 342)
    num_classes = 2
    
    # Training parameters
    batch_size = 32
    epochs = 1
    threshold = 48
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    
    # shuffle the patient ids
    patient_ids = random.shuffle(patient_ids)
    
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
    from model_utilities.load_model import load_model, load_imputer
    outcome_model = load_model(model_folder)
    imputer = load_imputer(model_folder)

    return outcome_model, imputer

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    # batch size has to be 1 because only one data per run
    from model_utilities.run_challenge_model import run_challenge_model_with_imputer
    outcome_pred, outcome_probability, cpc = run_challenge_model_with_imputer(models, data_folder, patient_id)

    return outcome_pred, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################
