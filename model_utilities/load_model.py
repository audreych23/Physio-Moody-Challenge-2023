import os 
import tensorflow as tf
import joblib
import preview_utilities.graphing as plotter

def load_model(model_folder):
    foldername_outcome = os.path.join(model_folder, 'model_outcome')
    # foldername_cpc = os.path.join(model_folder, 'model_cpc')
    
    # cpc_model = tf.keras.models.load_model(foldername_cpc)
    # cpc_model.summary()
    outcome_model = tf.keras.models.load_model(foldername_outcome)
    outcome_model.summary()

    graph_folder = os.path.join(model_folder, "graph")
    os.makedirs(graph_folder, exist_ok=True)

    plotter.plot_model(outcome_model, graph_folder, 'model_architecture.png')
    return outcome_model

def load_imputer(model_folder):
    imputer_filename = os.path.join(model_folder, 'model_outcome', 'models.sav')
    return joblib.load(imputer_filename)