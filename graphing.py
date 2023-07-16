#!/usr/bin/env python
from helper_code import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_train_curve():
    return

def plot_loss_curve():
    return

def plot_data_graph(list_patient_ids, data_folder):
    """Plot the label data (cpc data) with the count of the data

    Args: 
        list_patient_ids: the list of patient ids
        data_folder: The folder that contains the data
    """
    dict_cpc_count = dict()
    for patient_id in list_patient_ids:
        patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)
        cpc = get_cpc(patient_metadata)
        if cpc not in dict_cpc_count:
            dict_cpc_count[cpc] = 1
        else:
            dict_cpc_count[cpc] += 1

    df = pd.DataFrame({
        'cpc': list(dict_cpc_count.keys()),
        'count': list(dict_cpc_count.values()),
    })
    sns.barplot(data=df, x='cpc', y='count')
    plt.show()