from helper_code import *
import numpy as np, os, sys
import mne
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from model_training.evaluation_metrics import *
import tensorflow as tf
import data_loading.data_generator as dg
import time 

# This file is anything related to training
# loss calculation
def loss(model, loss_fn, x, y_true, training):
    """Calculate loss between the predicted y value from the model and the actual y with the loss function passed 

        Args:
            model: The model that are used to predict y prediction
            loss_fn: The loss_fn specified, examples: tf.keras.losses.SparseCategoricalCrossentropy
            x: the x data
            y_true: actual y value 
            training: A boolean to specify if it is currently in training, it is needed for the model if there are layers with different behavior
                        during training versus inference (e.g dropout)
        
        Returns:
            the loss calculated with the specified loss function
    """
    y_pred = model(x, training=training)

    return loss_fn(y_true=y_true, y_pred=y_pred)

# calculate gradients to optimize model (back propagate)
def grad(model, loss_fn, inputs, targets):
    """Calculate loss between the predicted y value from the model and the actual y with the loss function passed 

        Args:
            model: The model that are used to predict y prediction
            loss_fn: The loss_fn specified, examples: tf.keras.losses.SparseCategoricalCrossentropy
            inputs: the x data
            targets: actual y value

        Returns:
            A tuple of the loss value calculated and the gradient of all the model's trainable variables w.r.t loss
    """
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation.
    with tf.GradientTape() as tape:
        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.
        loss_value = loss(model, loss_fn, inputs, targets, training=True)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def custom_fit(graph_folder, model, num_epochs, training_data_gen, validation_data_gen=None):
    # Compile parameters
    learning_rate = 0.0001 
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # Metrics parameter
    train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    f1_accuracy_metric = tf.keras.metrics.F1Score()
    # Progress bar parameters
    metrics_names = ['acc','loss', 'val_acc', 'val_loss', 'f1_acc', 'challenge_score', 'AUROC', 'AUPRC']

    # Results (plotting)
    train_loss_results = []
    train_accuracy_results = []

    val_loss_results = []
    val_accuracy_results = []
    f1_score_results = []
    challenge_score_results = []
    auroc_results = []
    auprc_results = []

    # Dummy dictinary that is similar to callbacks
    history = dict()

    for epoch in range(num_epochs):
        train_loss_avg = tf.keras.metrics.Mean()
        val_loss_avg = tf.keras.metrics.Mean()

        print("\nepoch {}/{}".format(epoch+1,num_epochs))
    
        prog_bar_epoch = tf.keras.utils.Progbar(len(training_data_gen), stateful_metrics=metrics_names)

        # Training loop - using batches of batch_size 
        # x_batch dim : (batch_size, *dim(x)), y_batch dim : (batch_size, *dim(y))
        for x_batch_train, y_batch_train in training_data_gen:
            # print("x_batch shape:", np.shape(x_batch), ", y_batch shape:", np.shape(y_batch))
            time.sleep(0.3)
            # Optimize the model (forward pass and back propagate)
            loss_value, grads = grad(model, loss_fn, x_batch_train, y_batch_train)
            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            train_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            train_accuracy_metric.update_state(y_batch_train, model(x_batch_train, training=True))

            # Update progress bar every step (after batch_size batch is finished processing)
            prog_bar_values=[('acc', np.array(train_accuracy_metric.result())), ('loss', np.array(train_loss_avg.result()))]
        
            prog_bar_epoch.add(1, values=prog_bar_values)

        # End epoch
        train_loss_results.append(train_loss_avg.result())
        train_accuracy_results.append(train_accuracy_metric.result())
        # Reset training metrics at the end of each epoch
        train_accuracy_metric.reset_states()
        train_loss_avg.reset_state()

        # Create Probability Model
        softmax_layer = tf.keras.layers.Softmax()(model.output)
        probability_model = tf.keras.models.Model(inputs=model.input, outputs=softmax_layer)

        list_probability_outcome = list()
        list_true_outcome = list()
        list_pred_outcome = list()

        # Run a validation loop at the end of each epoch.
        if validation_data_gen is not None:
            for x_batch_val, y_batch_val in validation_data_gen:
                loss_value = loss(model, loss_fn, x_batch_val, y_batch_val, training=False)
                # Update val metrics
                y_batch_outcome_pred_prob = np.array(probability_model(x_batch_val, training=False))
                y_batch_outcome_pred = np.argmax(y_batch_outcome_pred_prob, axis=1).astype(np.int64)
                y_batch_outcome_pred = np.reshape(y_batch_outcome_pred, (-1, 1))

                val_accuracy_metric.update_state(y_batch_val, model(x_batch_val, training=False))
                val_loss_avg.update_state(loss_value)
                f1_accuracy_metric.update_state(y_batch_val, y_batch_outcome_pred)
                print(y_batch_outcome_pred_prob)
                y_batch_outcome_pred = np.argmax(y_batch_outcome_pred_prob, axis=1).astype(np.int64)
                y_batch_outcome_pred_prob = np.array(y_batch_outcome_pred_prob[:, 1])

                # dumb way to make an array of list
                for i, y in enumerate(y_batch_val):
                    list_probability_outcome.append(round(y_batch_outcome_pred_prob[i], 3))
                    list_pred_outcome.append(y_batch_outcome_pred[i])
                    list_true_outcome.append(y[0])
                    print(y)
                    print(y_batch_outcome_pred_prob[i])
                    print(y_batch_outcome_pred[i])

            
            list_probability_outcome = np.array(list_probability_outcome)
            list_pred_outcome = np.array(list_pred_outcome)
            list_true_outcome = np.array(list_true_outcome)

            print(list_true_outcome)
            print(list_probability_outcome)
            
            challenge_score = compute_challenge_score(list_true_outcome, list_probability_outcome)
            auroc_outcomes, auprc_outcomes = compute_auc(list_true_outcome, list_probability_outcome)

            val_accuracy_results.append(val_accuracy_metric.result())
            val_loss_results.append(val_loss_avg.result())
            f1_score_results.append(f1_accuracy_metric.result())
            challenge_score_results.append(challenge_score)
            auroc_results.append(auroc_outcomes)
            auprc_results.append(auprc_outcomes)
            print(challenge_score)
            print(auroc_outcomes)
            print(auprc_outcomes)

            val_accuracy_metric.reset_states()
            val_loss_avg.reset_states()
            f1_accuracy_metric.reset_states()

            prog_bar_values=[('acc', np.array(train_accuracy_results[-1])), ('loss', np.array(train_loss_results[-1])), 
                            ('val_acc', np.array(val_accuracy_results[-1])), ('val_loss', np.array(val_loss_results[-1])),
                            ('f1_score', np.array(f1_score_results[-1])), ('challenge_score', np.array(challenge_score_results[-1])),
                            ('AUROC', np.array(auroc_results[-1])), ('AUPRC', np.array(auprc_results[-1]))]
            
            
            prog_bar_epoch.add(0, values=prog_bar_values)

    
    # Some simple trade off dumb stuff you can do ;D
    history['accuracy'] = train_accuracy_results
    history['loss'] = train_loss_results
    if validation_data_gen is not None:
        history['val_accuracy'] = val_accuracy_results
        history['val_loss'] = val_loss_results
        # plot in last epoch
        make_roc_graph(list_true_outcome, list_probability_outcome, graph_folder=graph_folder)
        plot_confusion_matrix(list_true_outcome, list_pred_outcome, graph_folder=graph_folder)
        plot_confusion_matrix_challenge_score(list_true_outcome, list_probability_outcome, graph_folder=graph_folder)

    return history
    # history['val_accuracy'] = None
    # history['val_loss'] = None
        # if epoch % 1 == 0:
        #     print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
        #                                                                 epoch_loss_avg.result(),
        #                                                                 epoch_accuracy.result()))
        