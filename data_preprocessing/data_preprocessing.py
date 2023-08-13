import tensorflow as tf
import numpy as np
import helper_code as hp
from sklearn.impute import SimpleImputer

def impute_clinical_data(missing_values, strategy):
    imputer = SimpleImputer(missing_values=missing_values, strategy=strategy)
    imputer.fit()
    return imputer
