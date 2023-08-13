import tensorflow as tf

def model_lstm(timesteps, features_shape, num_classes):
    # delta model
    # timesteps x features
    assert features_shape == (342, 180, 180, 828)
    delta_inputs = tf.keras.layers.Input(
        shape=(timesteps, features_shape[0])
    )
    delta_l1 = tf.keras.layers.Masking(mask_value=0.)(delta_inputs)
    delta_l2 = tf.keras.layers.LSTM(128)(delta_l1)
    delta_l3 = tf.keras.layers.Dense(32, activation='relu')(delta_l2)

    # theta model
    theta_inputs = tf.keras.layers.Input(
        shape=(timesteps, features_shape[1])
    )
    theta_l1 = tf.keras.layers.Masking(mask_value=0.)(theta_inputs)
    theta_l2 = tf.keras.layers.LSTM(128)(theta_l1)
    theta_l3 = tf.keras.layers.Dense(16, activation='relu')(theta_l2)

    # alpha model
    alpha_inputs = tf.keras.layers.Input(
        shape=(timesteps, features_shape[2])
    )
    alpha_l1 = tf.keras.layers.Masking(mask_value=0.)(alpha_inputs)
    alpha_l2 = tf.keras.layers.LSTM(128)(alpha_l1)
    alpha_l3 = tf.keras.layers.Dense(16, activation='relu')(alpha_l2)

    # beta model
    beta_inputs = tf.keras.layers.Input(
        shape=(timesteps, features_shape[3])
    )
    beta_l1 = tf.keras.layers.Masking(mask_value=0.)(beta_inputs)
    beta_l2 = tf.keras.layers.LSTM(128)(beta_l1)
    beta_l3 = tf.keras.layers.Dense(32, activation='relu')(beta_l2)
    
    # Merge all the models
    concatenated_layers = tf.keras.layers.concatenate([delta_l3, theta_l3, alpha_l3, beta_l3])

    concatenated_l4 = tf.keras.layers.Dense(32, activation='relu')(concatenated_layers)
    output_layer = tf.keras.layers.Dense(num_classes)(concatenated_l4)

    merged_model = tf.keras.models.Model(inputs=[(delta_inputs, theta_inputs, alpha_inputs, beta_inputs)], outputs=[output_layer])
    merged_model.summary()
    return merged_model

def model_lstm_clinical_data(timesteps, features_shape, num_classes):
    # delta model
    # timesteps x features
    assert features_shape == (1530, 8)
    timeseries_inputs = tf.keras.layers.Input(
        shape=(timesteps, features_shape[0])
    )
    timeseries_l1 = tf.keras.layers.Masking(mask_value=0.)(timeseries_inputs)
    timeseries_l2 = tf.keras.layers.LSTM(32)(timeseries_l1)
    timeseries_l3 = tf.keras.layers.Dense(8, activation='relu')(timeseries_l2)

    # theta model
    clinical_inputs = tf.keras.layers.Input(
        shape=(features_shape[1])
    )
    clinical_l2 = tf.keras.layers.Dense(128, activation='relu')(clinical_inputs)

    # Merge all the models
    concatenated_layers = tf.keras.layers.concatenate([timeseries_l3, clinical_l2])

    concatenated_l4 = tf.keras.layers.Dense(4, activation='relu')(concatenated_layers)
    output_layer = tf.keras.layers.Dense(num_classes)(concatenated_l4)

    merged_model = tf.keras.models.Model(inputs=[(timeseries_inputs, clinical_inputs)], outputs=[output_layer])

    merged_model.summary()
    return merged_model

def model_eeg(
    input_channels:int,
    time_points:int,
    output_channels:int,
    n_estimators:int=8,
    dw_filters:int=8,
    cn_filters:int=16,
    sp_filters:int=32
):  
    """ EEGNet + RNN Model.
    input_channels: number of data's channel
    time_points: cumulated data's total time point
    output_channels: number of output node
    n_estimators: number of RNN cells
    dw_filters: number of filters for DepthwiseConv2D
    cn_filters: number of filters for Conv2D (1st filter from the paper EEGNet)
    sp_filters: number of filters for SeparableConv2D (2nd filter from the paper EEGNet)7\
    
    return  model: tf.keras.Model
    """
    # Define input layer
    inputs = tf.keras.layers.Input(
        shape=(
            input_channels, 
            time_points,
        )
    )
    cn = tf.keras.layers.Reshape((input_channels, time_points, 1))(inputs)
    cn = tf.keras.layers.Conv2D(filters=cn_filters, 
            kernel_size=(1, 64), 
            padding='same', 
            activation='linear',
            use_bias=False,
            )(cn)
    cn = tf.keras.layers.BatchNormalization()(cn)

    # Depthwise convolution layer
    dw = tf.keras.layers.DepthwiseConv2D(kernel_size=(input_channels, 1),
                        padding='valid',
                        depth_multiplier=dw_filters,
                        depthwise_constraint=tf.keras.constraints.max_norm(1.),
                        activation='linear',
                        use_bias=False,
                        )(cn)
    dw = tf.keras.layers.BatchNormalization()(dw)
    dw = tf.keras.layers.Activation('elu')(dw)
    dw = tf.keras.layers.AveragePooling2D(pool_size=(1, 4),
                        padding='valid',
                        )(dw)
    
    # Separable convolution layer
    sp = tf.keras.layers.SeparableConv2D(filters=sp_filters,
                        kernel_size=(1, 8),
                        padding='same',
                        activation='linear',
                        use_bias=False,
                        )(dw)
    sp = tf.keras.layers.BatchNormalization()(sp)
    sp = tf.keras.layers.Activation('elu')(sp)

    # # RNN layer
    # shape = tuple([x for x in sp.shape.as_list() if x != 1 and x is not None])
    # sp = tf.keras.layers.Reshape(shape)(sp)
    # sp = tf.keras.layers.GRU(n_estimators, return_sequences=True)(sp)
    # sp = tf.keras.layers.Dropout(0.5)(sp)

    # Flatten output
    sp = tf.keras.layers.Flatten()(sp)

    # Output layer
    outputs = tf.keras.layers.Dense(output_channels,
                    kernel_constraint=tf.keras.constraints.max_norm(0.25),
                    )(sp)
    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model