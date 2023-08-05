from helper_code import *
import numpy as np, os, sys
import mne
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import tensorflow as tf
import data_generator as dg
import time 

# This file is anything related to training
# loss calculation
def loss(model, loss_fn, x, y_true, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_pred = model(x, training=training)

  return loss_fn(y_true=y_true, y_pred=y_pred)

# calculate gradients to optimize model (back propagate)
def grad(model, loss_fn, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, loss_fn, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def model():
   return 

def custom_fit(model, features, labels):
    # Train parameters
    learning_rate = 0.0001
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    batch_size = 16

    # Generator
    ds_train_batch = None
    
    num_epochs = 20
    # Progress bar parameters
    metrics_names = ['acc','loss']

    # Definition
    # dummy patient_ids
    patient_ids = range(300)
    num_training_samples = len(patient_ids)
    
    # Results (plotting)
    train_loss_results = []
    train_accuracy_results = []


    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        print("\nepoch {}/{}".format(epoch+1,num_epochs))
    
        prog_bar_epoch = tf.keras.utils.Progbar(num_training_samples, stateful_metrics=metrics_names)

        # Training loop - using batches of batch_size 
        # x_batch dim : (batch_size, *dim(x)), y_batch dim : (batch_size, *dim(y))
        for x_batch, y_batch in ds_train_batch:
            time.sleep(0.3)
            # Optimize the model
            loss_value, grads = grad(model, loss_fn, x_batch, y_batch)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(y_batch, model(x_batch, training=True))

            prog_bar_values=[('acc', np.array(epoch_accuracy.result())), ('pr', np.array(epoch_loss_avg.result()))]
        
            prog_bar_epoch.add(batch_size, values=prog_bar_values)

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        # if epoch % 1 == 0:
        #     print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
        #                                                                 epoch_loss_avg.result(),
        #                                                                 epoch_accuracy.result()))
        