#!/usr/bin/env python
from helper_code import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

    df = pd.DataFrame({
        'cpc': list(dict_cpc_count.keys()),
        'count': list(dict_cpc_count.values()),
    })
    sns.barplot(data=df, x='cpc', y='count')
    plt.savefig(os.path.join(graph_folder, "data_cpc_count.png"))
    
    df = pd.DataFrame({
        'outcome': list(dict_outcome_count.keys()),
        'count': list(dict_outcome_count.values()),
    })

    sns.barplot(data=df, x='outcome', y='count')
    plt.savefig(os.path.join(graph_folder, "data_outcome_count.png"))
    # Plot binary good (1) and bad (0) with count
    
    # for cpc in dict_cpc_count:
    #     # cpc 1, 2 is good
    #     if cpc < 3:
    #         if 'good : 0' not in dict_outcome_count:
    #             dict_outcome_count['good : 0'] = dict_cpc_count[cpc] 
    #         else: 
    #             dict_outcome_count['good : 0'] += dict_cpc_count[cpc] 
    #     # cpc 3, 4, 5 is baf
    #     else:
    #         if 'poor : 1' not in dict_outcome_count:
    #             dict_outcome_count['poor : 1'] = dict_cpc_count[cpc]
    #         else:
    #             dict_outcome_count['poor : 1'] += dict_cpc_count[cpc]
    
    # df = pd.DataFrame({
    #     'outcome': list(dict_outcome_count.keys()),
    #     'count': list(dict_outcome_count.values()),
    # })

    # sns.barplot(data=df, x='outcome', y='count')
    # plt.savefig("data_outcome_count")
    # plt.show()