#!/usr/bin/env python
from helper_code import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_roc_graph(tpr, fpr, graph_folder):
    # tpr is also called sensitivity
    # fpr is 1 - specificity
    # x-axis : tpr, y-axis : fpr
    # threshold is fixed at 0.05
    x_fpr_threshold = [0.05, 0.05]
    y_fpr_threshold = [0, 1]
    plt.figure()
    plt.plot(fpr, tpr, label='roc', marker='.')
    plt.plot(x_fpr_threshold, y_fpr_threshold, label='fpr 0.05 line', linestyle='--')
    plt.yticks(np.arange(min(tpr), max(tpr)+0.05, 0.05))
    plt.legend()
    plt.savefig(os.path.join(graph_folder, "roc_graph.png"))

    return

def plot_accuracy_curve(history, graph_folder):
    plt.figure()
    plt.plot(history.history['accuracy'])
    if ('val_accuracy' in history.history):
        plt.plot(history.history['val_accuracy'])
    plt.title('model_accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    if ('val_accuracy' in history.history):
        plt.legend(['train', 'val'], loc='upper left')
    else:
        plt.legend(['train'], loc='upper left')
    plt.savefig(os.path.join(graph_folder, "accuracy_curve.png"))
    return

def plot_loss_curve(history, graph_folder):
    plt.figure()
    plt.plot(history.history['loss'])
    if ('val_loss' in history.history):
        plt.plot(history.history['val_loss'])
    plt.title('model_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    if ('val_loss' in history.history):
        plt.legend(['train', 'val'], loc='upper left')
    else:
        plt.legend(['train'], loc='upper left')
    plt.savefig(os.path.join(graph_folder, "loss_curve.png"))
    return


def plot_data_graph(list_patient_ids, data_folder, graph_folder):
    """Plot the label data (cpc data) with the count of the data

    Args: 
        list_patient_ids: the list of patient ids
        data_folder: The folder that contains the data
    """
    
    # plot cpc with count graph
    
    dict_cpc_count = dict()
    dict_outcome_count = dict()
    for patient_id in list_patient_ids:
        patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)
        cpc = get_cpc(patient_metadata)
        outcome = get_outcome(patient_metadata)
        if cpc not in dict_cpc_count:
            dict_cpc_count[cpc] = 1
        else:
            dict_cpc_count[cpc] += 1
        
        if outcome not in dict_outcome_count:
            dict_outcome_count[outcome] = 1
        else:
            dict_outcome_count[outcome] += 1
            
    plt.figure()
    df = pd.DataFrame({
        'cpc': list(dict_cpc_count.keys()),
        'count': list(dict_cpc_count.values()),
    })
    
    ax = sns.barplot(data=df, x='cpc', y='count')
    for i in ax.containers:
        ax.bar_label(i,)
    plt.savefig(os.path.join(graph_folder, "data_cpc_count.png"))
    
    plt.figure()
    df = pd.DataFrame({
        'outcome': list(dict_outcome_count.keys()),
        'count': list(dict_outcome_count.values()),
    })

    ax = sns.barplot(data=df, x='outcome', y='count')
    for i in ax.containers:
        ax.bar_label(i,)
    plt.savefig(os.path.join(graph_folder, "data_outcome_count.png"))
