# call this if you want to check for cross validation
def k_fold_cross_validation(available_signal_datas, outcomes, cpcs, prefix_sum_index, num_patients, outcomes_random_forest):
    validation_accuracies_outcome = list()
    validation_losses_outcome = list()

    validation_accuracies_cpc = list()
    validation_losses_cpc = list()
    print_flag = 1
    kf = KFold(n_splits=5)
    # dummy variable to split the train and val_index
    for i, (train_index, val_index) in enumerate(kf.split(np.zeros((num_patients, 1)), outcomes_random_forest)):
        # Split the data into train and validation with k-fold cross validation
        if (print_flag == 1):
            print("train_index, val_index", train_index, val_index)
        # prepare x data for training and validation
        x_train = list()
        x_val = list()

        # prepare y data for training and validation
        y_train_outcome = list()
        y_val_outcome = list()
        y_train_cpc = list()
        y_val_cpc = list()
        # equivalent on using iloc
        for idx in val_index:
            start = prefix_sum_index[idx]
            end = prefix_sum_index[idx + 1]
            # use the matrix from [start: end] and concatenate with the next start, next end
            x_val.append(available_signal_datas[start:end])
            y_val_outcome.append(outcomes[start:end])
            y_val_cpc.append(cpcs[start:end])

        for idx in train_index:
            start = prefix_sum_index[idx]
            end = prefix_sum_index[idx + 1]
            # use the matrix from [start: end] and concatenate with the next start, next end
            x_train.append(available_signal_datas[start:end])
            y_train_outcome.append(outcomes[start:end])
            y_train_cpc.append(cpcs[start:end])
        
        # maybe this is why it broke
        x_val = np.vstack(x_val)
        x_train = np.vstack(x_train)
        y_train_outcome = np.vstack(y_train_outcome)
        y_val_outcome = np.vstack(y_val_outcome)
        y_train_cpc = np.vstack(y_train_cpc)
        y_val_cpc = np.vstack(y_val_cpc)

        if (print_flag == 1):
            # sanity check
            print("x_val", x_val.shape)
            print("x_train", x_train.shape)
            print("y_train_outcome", y_train_outcome.shape)
            print("y_val_outcome", y_val_outcome.shape)
            print("y_train_cpc", y_train_cpc.shape)
            print("y_val_cpc", y_val_cpc.shape)

        # Train the models.
        if print_flag >= 1:
            print('Training the Challenge models on the Challenge data...')

        # unusable right now don't call this function !
        # model_lstm_outcome
        model_lstm_outcome = create_model_lstm(x_train, 0)
        model_lstm_outcome.summary()
        # model_lstm_outcome = compile_model(x_train, y_train_outcome, x_val, y_val_outcome, model_lstm_outcome, 0)

        # evaluate the model
        results = model_lstm_outcome.evaluate(x=x_val, y=y_val_outcome)
        results = dict(zip(model_lstm_outcome.metrics_names, results))
        
        validation_accuracies_outcome.append(results['accuracy'])
        validation_losses_outcome.append(results['loss'])
        print(i, "- outcome validation accuracy, loss", results['accuracy'], results['loss'])

        # model_lstm_cpc
        model_lstm_cpc = create_model_lstm(x_train, 1)
        model_lstm_cpc.summary()
        # model_lstm_cpc = compile_model(x_train, y_train_cpc, x_val, y_val_cpc, model_lstm_cpc, 1)

        results = model_lstm_cpc.evaluate(x=x_val, y=y_val_cpc)
        results = dict(zip(model_lstm_cpc.metrics_names, results))

        validation_accuracies_cpc.append(results['accuracy'])
        validation_losses_cpc.append(results['loss'])

        tf.keras.backend.clear_session()
        if print_flag >= 1:
            print("Done", i, "training")

    for i, v in enumerate(validation_accuracies_outcome):
        print(i, '- (accuracy, losses) = ', validation_accuracies_outcome[i], validation_losses_outcome[i])

    avg_accuracy_outcome = average(validation_accuracies_outcome)
    avg_loss_outcome = average(validation_losses_outcome)
    print("avg accuracy:", avg_accuracy_outcome, ", avg loss:", avg_loss_outcome)

    for i, v in enumerate(validation_accuracies_cpc):
        print(i, '- (accuracy, losses) = ', validation_accuracies_cpc[i], validation_losses_cpc[i])

    avg_accuracy_outcome = average(validation_accuracies_outcome)
    avg_loss_outcome = average(validation_losses_outcome)
    print("avg accuracy:", avg_accuracy_outcome, ", avg loss:", avg_loss_outcome)