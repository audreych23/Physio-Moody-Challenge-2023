import os 
import tensorflow as tf
import joblib

def load_model(model_folder):
    foldername_outcome = os.path.join(model_folder, 'model_outcome')
    # foldername_cpc = os.path.join(model_folder, 'model_cpc')
    
    # cpc_model = tf.keras.models.load_model(foldername_cpc)
    # cpc_model.summary()
    outcome_model = tf.keras.models.load_model(foldername_outcome)
    outcome_model.summary()

    graph_folder = os.path.join(model_folder, "graph")
    os.makedirs(graph_folder, exist_ok=True)
    model_img_filename = os.path.join(graph_folder, 'model_architecture.png')

    tf.keras.utils.plot_model(outcome_model, to_file=model_img_filename, show_shapes=True)
    
    return outcome_model

def load_imputer(model_folder):
    imputer_filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(imputer_filename), 