from helper_code import *
import numpy as np, os, sys
import mne
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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

def custom_fit(model, num_epochs, training_data_gen, validation_data_gen=None):
    # Compile parameters
    learning_rate = 0.0001 
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # Metrics parameter
    train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    # Progress bar parameters
    metrics_names = ['acc','loss', 'val_acc', 'val_loss']

    # Results (plotting)
    train_loss_results = []
    train_accuracy_results = []
    val_loss_results = []
    val_accuracy_results = []

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

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in validation_data_gen:
            loss_value = loss(model, loss_fn, x_batch_val, y_batch_val, training=False)
            # Update val metrics
            val_accuracy_metric.update_state(y_batch_val, model(x_batch_val, training=False))
            val_loss_avg.update_state(loss_value)

        val_accuracy_results.append(val_accuracy_metric.result())
        val_loss_results.append(val_loss_avg.result())
        val_accuracy_metric.reset_states()
        val_loss_avg.reset_states()

        prog_bar_values=[('acc', np.array(train_accuracy_results[-1])), ('loss', np.array(train_loss_results[-1])), 
                         ('val_acc', np.array(val_accuracy_results[-1])), ('val_loss', np.array(val_loss_results[-1]))]
        
        prog_bar_epoch.add(0, values=prog_bar_values)

    # Some simple trade off dumb stuff you can do ;D
    history['accuracy'] = train_accuracy_results
    history['loss'] = train_loss_results
    history['val_accuracy'] = val_accuracy_results
    history['val_loss'] = val_loss_results

    return history
    # history['val_accuracy'] = None
    # history['val_loss'] = None
        # if epoch % 1 == 0:
        #     print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
        #                                                                 epoch_loss_avg.result(),
        #                                                                 epoch_accuracy.result()))
        