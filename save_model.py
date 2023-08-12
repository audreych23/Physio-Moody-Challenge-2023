import os
import joblib
# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)

def save_challenge_model_lstm(model_folder, model, folder_name):
    os.makedirs(os.path.join(model_folder, folder_name), exist_ok=True)
    model.save(os.path.join(model_folder, folder_name))
