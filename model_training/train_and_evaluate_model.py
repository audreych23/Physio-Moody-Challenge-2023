import numpy as np
from helper_code import *
import numpy as np, os, sys
import mne
from data_preprocessing.data_preprocessing import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import data_loading.data_generator as dg
from models.models import *
import preview_utilities.graphing as plotter
from model_training.custom_train import *
from model_utilities.save_model import *
from data_preprocessing.preload_clinical_data import *
import preview_utilities.evaluation_writer as evaluation_writer

# call this if you want to check for cross validation
def k_fold_cross_validation(data_folder, model_folder, graph_folder, verbose, patient_ids, 
                            timesteps, features_dim, num_classes, batch_size, epochs, k = 5):
    
    if k <= 1:
        raise Exception('Input k has to be more than 1')

    total_num_patients_train = len(patient_ids)

    num_patients_validation = int(round(total_num_patients_train * (1 / k)))

    for idx in range(k):
        print('idx:', idx)
        # edge case, last of iteration, take the remaining data
        if idx == k - 1:
            patient_ids_val = patient_ids[idx * num_patients_validation:]
        else:    
            patient_ids_val = patient_ids[idx * num_patients_validation: (idx + 1) * num_patients_validation]
        
        # i.e. k = 5 and n = 8
        if len(patient_ids_val) == 0:
            print('Warning: amount of data folds are not the same as k, please consider changing the number of folds')
            continue

        patient_ids_train = patient_ids[: idx * num_patients_validation] + patient_ids[(idx + 1) * num_patients_validation: ]


        print('Preload clinical data to train imputer...')
        # preload all clinical data to get the imputer (not sure if we can do streaming on it)
        preloaded_patient_features = preload_clinical_data(patient_ids_train, data_folder)
        clinical_data_imputer = create_clinical_data_imputer(missing_values=np.nan, strategy='mean', x_data=preloaded_patient_features)

        # Extract the features and labels.
        if verbose >= 1:
            print('Extracting features and labels from the Challenge data...')

            # Create Data Generator
        if verbose >= 1:
            print('Creating data generator...')

        training_generator = dg.DataGenerator(patient_ids_train, data_folder, batch_size=batch_size, imputer=clinical_data_imputer)
        validation_generator = dg.DataGenerator(patient_ids_val, data_folder, batch_size=batch_size, imputer=clinical_data_imputer)

        # Create Model
        model_outcome = model_lstm_clinical_data_separate(timesteps, features_dim, num_classes)

        # Train Model
        if verbose >= 1:
            print('Training the Challenge models on the Challenge data...')

        # output of a custom fit is dictionary (for now)
        history_outcome, list_probability_outcome, list_pred_outcome, list_true_outcome = custom_fit(graph_folder, model_outcome, epochs, training_generator, validation_generator)

        # Plot graph
        plotter.plot_loss_curve_dict(history_outcome, graph_folder, f'loss_curve_{idx}.png')
        plotter.plot_accuracy_curve_dict(history_outcome, graph_folder, f'accuracy_curve_{idx}.png')

        # plot other curves in last epoch
        make_roc_graph(list_true_outcome, list_probability_outcome, graph_folder=graph_folder, graph_name=f'roc_graph_{idx}.png')
        plot_confusion_matrix(list_true_outcome, list_pred_outcome, graph_folder=graph_folder, graph_name=f'confusion_matrix_{idx}.png')
        plot_confusion_matrix_challenge_score(list_true_outcome, list_probability_outcome, graph_folder=graph_folder, graph_name=f'confusion_matrix_challenge_{idx}.png')

        # Write result of every epoch and k on a csv file
        evaluation_writer.write_evaluation_result(graph_folder, f'scores_{idx}.csv', history_outcome)

        # Save model
        save_challenge_model_lstm(model_folder, model_outcome, clinical_data_imputer, f"model_outcome_{idx}")
        if verbose >= 1:
            print(f'Done. idx = {idx}')

    return
            

def train_and_evaluate_model(data_folder, model_folder, graph_folder, verbose, patient_ids, 
                            timesteps, features_dim, num_classes, batch_size, epochs, validation, data_split_validation):

        # Extract the features and labels.
        if verbose >= 1:
            print('Extracting features and labels from the Challenge data...')

        # Create Data Generator
        if verbose >= 1:
            print('Creating data generator...')

        if not validation:
            # Train whole data if no validaiton
            # preload all clinical data to get the imputer (not sure if we can do streaming on it)
            preloaded_patient_features = preload_clinical_data(patient_ids, data_folder)
            clinical_data_imputer = create_clinical_data_imputer(missing_values=np.nan, strategy='mean', x_data=preloaded_patient_features)

            training_generator = dg.DataGenerator(patient_ids, data_folder, batch_size=batch_size, imputer=clinical_data_imputer)

        else:
            total_num_patients_train = len(patient_ids)

            num_patients_validation = np.ceil(total_num_patients_train * (data_split_validation / 100)).astype(np.int32)
            num_patients_train = total_num_patients_train - num_patients_validation

            patient_ids_train = patient_ids[:num_patients_train]
            patient_ids_val = patient_ids[num_patients_train:]

            # preload all clinical data to get the imputer (not sure if we can do streaming on it)
            preloaded_patient_features = preload_clinical_data(patient_ids_train, data_folder)
            clinical_data_imputer = create_clinical_data_imputer(missing_values=np.nan, strategy='mean', x_data=preloaded_patient_features)

            training_generator = dg.DataGenerator(patient_ids_train, data_folder, batch_size=batch_size, imputer=clinical_data_imputer)
            validation_generator = dg.DataGenerator(patient_ids_val, data_folder, batch_size=batch_size, imputer=clinical_data_imputer)

        # Create Model
        model_outcome = model_lstm_clinical_data_separate(timesteps, features_dim, num_classes)

        # Train Model
        if verbose >= 1:
            print('Training the Challenge models on the Challenge data...')

        # output of a custom fit is dictionary (for now)
        if validation:
            history_outcome, list_probability_outcome, list_pred_outcome, list_true_outcome = custom_fit(graph_folder, model_outcome, epochs, training_generator, validation_generator)
            
            # plot other curves in last epoch
            make_roc_graph(list_true_outcome, list_probability_outcome, graph_folder=graph_folder, graph_name=f'roc_graph.png')
            plot_confusion_matrix(list_true_outcome, list_pred_outcome, graph_folder=graph_folder, graph_name=f'confusion_matrix.png')
            plot_confusion_matrix_challenge_score(list_true_outcome, list_probability_outcome, graph_folder=graph_folder, graph_name=f'confusion_matrix_challenge.png')

            # Write result of challenge score at every epoch
            evaluation_writer.write_evaluation_result(graph_folder, f'scores.csv', history_outcome)

        else:
            history_outcome = custom_fit(graph_folder, model_outcome, epochs, training_generator)

        # Plot graph
        plotter.plot_loss_curve_dict(history_outcome, graph_folder)
        plotter.plot_accuracy_curve_dict(history_outcome, graph_folder)

        # Save model
        save_challenge_model_lstm(model_folder, model_outcome, clinical_data_imputer, "model_outcome")
        if verbose >= 1:
            print('Done.')