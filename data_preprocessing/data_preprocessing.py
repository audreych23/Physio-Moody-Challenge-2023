import tensorflow as tf
import numpy as np
import helper_code as hp
from sklearn.impute import SimpleImputer

def create_clinical_data_imputer(missing_values, strategy, x_data):
    imputer = SimpleImputer(missing_values=missing_values, strategy=strategy)
    imputer.fit(x_data)
    return imputer
