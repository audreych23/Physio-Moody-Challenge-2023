import tensorflow as tf
import data_loading.data_generator as dg
import numpy as np

def run_challenge_model_no_imputer(models, data_folder, patient_id):
    outcome_model = models

    softmax_layer = tf.keras.layers.Softmax()(outcome_model.output)
    probability_model = tf.keras.models.Model(inputs=outcome_model.input, outputs=softmax_layer)
    patient_ids = list()
    patient_ids.append(patient_id)
    # Load data.
    test_data_generator = dg.DataGenerator(patient_ids, data_folder, batch_size=1, to_fit=False, shuffle=False, imputer=None)
        
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

def run_challenge_model_with_imputer(models, data_folder, patient_id):
    outcome_model = models[0]
    # cpc_model = models[1]
    imputer_model = models[1]
    imputer = imputer_model['imputer']
    # outcome_model = random_tree_model['outcome_model']
    # cpc_model = random_tree_model['cpc_model']

    softmax_layer = tf.keras.layers.Softmax()(outcome_model.output)
    probability_model = tf.keras.models.Model(inputs=outcome_model.input, outputs=softmax_layer)

    patient_ids = list()
    patient_ids.append(patient_id)
    # Load data.
    test_data_generator = dg.DataGenerator(patient_ids, data_folder, batch_size=1, to_fit=False, shuffle=False, imputer=imputer)
        
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