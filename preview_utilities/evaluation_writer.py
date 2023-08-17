import os

def write_evaluation_result(output_folder, output_name, history):
    # only training loss and acc is 2 
    assert len(history.items()) > 2
    result_path = os.path.join(output_folder, 'result')
    os.makedirs(result_path, exist_ok=True)

    # Unpack the scores.
    with open(os.path.join(result_path, output_name), 'w') as f:
        for i, _ in enumerate(history['f1_score']):
            challenge_score = history['challenge_score'][i] 
            auroc_outcomes = history['auroc'][i]
            auprc_outcomes = history['auprc'][i]
            # accuracy_outcomes = 
            f_measure_outcomes = history['f1_score'][i]
            # mse_cpcs, mae_cpcs = scores

            # Construct a string with scores.

            output_string = \
                f'Epoch: {i}\n' + \
                'Challenge Score: {:.3f}\n'.format(challenge_score) + \
                'Outcome AUROC: {:.3f}\n'.format(auroc_outcomes) + \
                'Outcome AUPRC: {:.3f}\n'.format(auprc_outcomes) + \
                'Outcome F-measure: {:.3f}\n'.format(f_measure_outcomes) + \
                '\n'
                # 'Outcome Accuracy: {:.3f}\n'.format(accuracy_outcomes) + \
                # 'CPC MSE: {:.3f}\n'.format(mse_cpcs) + \
                # 'CPC MAE: {:.3f}\n'.format(mae_cpcs)
            
            f.write(output_string)

    return