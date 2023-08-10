import os
import numpy as np
import tensorflow as tf
import helper_code as hp
import mne
# one think im worried about is the corner case when > batch_size
# patient_ids[0] store patient id eg '0284'
class DataGenerator(tf.keras.utils.Sequence):
    """This class is a data generator use to load partial data in memory, batch_size is recommended to be less than 4
        Dimension of the actual data depends on each patient hours data availability and batch_size
        i.e. if batch_size is 4, and each patient has 15, 32, 42, 52 hours then available_h_in_patient_per_batch_size = 15 + 32 + 42 + 52
    """
    def __init__(self, patient_ids, data_path,
               to_fit = True, batch_size = 8, threshold = 48,
               shuffle = True):
        """Constructor

        Args:
            patient_ids: List of patient ids
            data_path: The path to the data folder
            dim: Dimension of the data
            to_fit: Indicate if the data is used for training or evaluating
            batch_size: Batch Size for Training and Evaluating model
            shuffle: Indicate if the data will be shuffled after each epoch
        """
        # lets do something stupid by first taking data from only one lstm data which is the recentmost
        self.patient_ids = patient_ids
        self.patient_ids_index = np.arange(len(patient_ids))
        self.data_path = data_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.threshold = threshold
        self.on_epoch_end()
    
    def __len__(self):
        """Denotes the number of batches per epoch

        Returns: 
            number of batches per epoch
        """
        return int(np.ceil(len(self.patient_ids) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data

        Args:
            index: index of the batch
        """
        # Generate indexes of the batch
        # e.g. indexes[32 : 64]
        # corner case for high batch index where the last batch index is larger than the len of the data
        print('index:', index)
        low_batch_index = index * self.batch_size
        high_batch_index = (index + 1) * self.batch_size
        if (len(self.patient_ids_index) < high_batch_index):
            high_batch_index = len(self.patient_ids_index)

        # since the index is shuffled already so it is fine
        patient_ids_index_batch = self.patient_ids_index[low_batch_index: high_batch_index]

        # Find list of IDs
        patient_ids_batch = [self.patient_ids[idx] for idx in patient_ids_index_batch]
        # has a list of id ['0286', '0284', '0297', ...]
        # Generate data
        # dim of x data depends on batch size and available hours per patient
        delta_psd_data, theta_psd_data, alpha_psd_data, beta_psd_data = self._generate_x_data(patient_ids_batch)
        

        print("X data has finished generating...")
        # Sanity check if first dimension is batch size
        assert np.shape(delta_psd_data)[0] == len(patient_ids_batch) and np.shape(theta_psd_data)[0] == len(patient_ids_batch) \
                and np.shape(alpha_psd_data)[0] == len(patient_ids_batch) and np.shape(beta_psd_data)[0] == len(patient_ids_batch)
        
        if self.to_fit:
            y_data = self._generate_y_data(patient_ids_batch)
            print("y data has finished generating...")
            # print("shape of x and y data", np.shape(x_data), np.shape(y_data))
            assert np.shape(y_data)[0] == len(patient_ids_batch)

            return (delta_psd_data, theta_psd_data, alpha_psd_data, beta_psd_data), y_data
        else:
            return (delta_psd_data, theta_psd_data, alpha_psd_data, beta_psd_data), 
        
    def on_epoch_end(self):
        """Shuffle the indexes after each end of epoch

        """
        # get all the indexes of available patient id
        self.patient_ids_index = np.arange(len(self.patient_ids))
        if self.shuffle == True:
            # shuffle index 0 1 2 3 -> 3 1 0 2
            np.random.shuffle(self.patient_ids_index)

    def _generate_x_data(self, patient_ids_batch):
        """Generates x data of batch_size data

        Args:
            patient_ids_batch: Patient ids that have been shuffled and batched
            
        Returns:
            x_data: the data itself (dim: (batch_size x (*self.dim)))
            len_x_datas: list of the length of each data per patient in batch (useful to duplicate y_data) 
        """
        # Initialization

        list_delta_psd_data = list()
        list_theta_psd_data = list()
        list_alpha_psd_data = list()
        list_beta_psd_data = list()

        # Generate data
        # length of list_ids_temp should be according to batch_size
        print(patient_ids_batch)
        for i, patient_id in enumerate(patient_ids_batch):
            # Store sample
            print(i)
            print("wat", self.data_path)
            print(patient_id)
            patient_metadata, recording_metadata, recording_data = hp.load_challenge_data(self.data_path, patient_id)
            # just get most recent one - very simple
            patient_features, available_signal_data, delta_psd_data, theta_psd_data, alpha_psd_data, beta_psd_data = self._get_features(patient_metadata, recording_metadata, recording_data)
            
            delta_psd_data = self._arr_transformations_model(delta_psd_data)
            theta_psd_data = self._arr_transformations_model(theta_psd_data)
            alpha_psd_data = self._arr_transformations_model(alpha_psd_data)
            beta_psd_data = self._arr_transformations_model(beta_psd_data)
            # print("available_signal_data: ", np.shape(available_signal_data))
            list_delta_psd_data.append(delta_psd_data)
            list_theta_psd_data.append(theta_psd_data)
            list_alpha_psd_data.append(alpha_psd_data)
            list_beta_psd_data.append(beta_psd_data)
            # print(len(list_beta_psd_data))
            # x_data.append(available_signal_data)
            # len_x_datas.append(np.shape(available_signal_data)[0])
            # x_data[i,] = available_signal_data
        
        list_delta_psd_data = np.array(list_delta_psd_data, dtype=object)
        list_theta_psd_data = np.array(list_theta_psd_data, dtype=object)
        list_alpha_psd_data = np.array(list_alpha_psd_data, dtype=object)
        list_beta_psd_data = np.array(list_beta_psd_data, dtype=object)


        list_delta_psd_data = np.vstack(list_delta_psd_data)
        list_theta_psd_data = np.vstack(list_theta_psd_data)
        list_alpha_psd_data = np.vstack(list_alpha_psd_data)
        list_beta_psd_data = np.vstack(list_beta_psd_data)

        list_delta_psd_data = np.asarray(list_delta_psd_data).astype(np.float32)
        list_theta_psd_data = np.asarray(list_theta_psd_data).astype(np.float32)
        list_alpha_psd_data = np.asarray(list_alpha_psd_data).astype(np.float32)
        list_beta_psd_data = np.asarray(list_beta_psd_data).astype(np.float32)
  
        return list_delta_psd_data, list_theta_psd_data, list_alpha_psd_data, list_beta_psd_data

    def _arr_transformations_model(self, data_arr):
        """Convert from (X, Y, Z) into (X, Y * Z) into (72, Y * Z) into (1, 72, Y * Z) given X <= 72 to feed to the model
        """
        data_shape = np.shape(data_arr)
        data_arr = np.reshape(data_arr, (data_shape[0], data_shape[1] * data_shape[2]))
        data_arr = self._pad_timeseries_arr(data_arr)
        data_shape = np.shape(data_arr)
        data_arr = np.reshape(data_arr, (1, *data_shape))
        data_arr = np.asarray(data_arr).astype(np.float32)
        return data_arr

    # Pad 2 dimensions array
    def _pad_timeseries_arr(self, data_arr, desired_time_pad=72):
        current_shape = np.shape(data_arr)
        rows_to_pad = desired_time_pad - current_shape[0]
        padded_arr = np.pad(data_arr, ((0, rows_to_pad), (0, 0)), mode='constant')
        return padded_arr
    
    def _generate_y_data(self, patient_ids_batch):
        """Generate y data of batch_size data

        Args:
            patient_ids_batch: Patient ids that have been shuffled and batched
            num_classes: Number of classes for one hot encoding 
            len_x_datas: a list storing the length of each patient available (signal) hours 
        Returns:
            y_data: The label of the data (y data) in one hot (dim: (sum(len_x_datas) x num_classes)) (depends on how many hours is it available per patients)
        """
        # batch_size = len(patient_ids_batch)
        y_data = list()
        # Generate data
        for i, patient_id in enumerate(patient_ids_batch):
            # Store sample
            patient_metadata, recording_metadata, recording_data = hp.load_challenge_data(self.data_path, patient_id)

            outcome_data = hp.get_outcome(patient_metadata)
            y_data.append(outcome_data)

        y_data = np.array(y_data)
        y_data = np.vstack(y_data)
        y_data = np.asarray(y_data).astype(np.int32)

        return y_data
    
    def _duplicate_y_data(self, len_x_data, y_data):
        """This is needed for this specific to get the output for every time hour
        Args:
            len_x_data_batch : amount of y data that should be duplicated and it depends to how many x data have we added 
        Returns:
            patient_y_datas : the duplicated y_datas
        """
        patient_y_datas = list()
        for _ in range(len_x_data):
            patient_y_datas.append(y_data)
        # print(len(patient_y_datas))
        patient_y_datas = np.array(patient_y_datas)
        patient_y_datas = np.reshape(patient_y_datas, (-1, 1))
        return patient_y_datas
    
    def _get_features(self, patient_metadata, recording_metadata, recording_data):
        """Get the Timestamps Data.
        Args:
            patient_metadata (list): Data of Patient
            recording_metadata (list): Data of Recording
            recording_data (list): Recording Data
            threshold: threshold of hours, only take the data when hours is above threshold
            option (int):
                * 1: Recording timestamp with PSDs.
                * 2: Recording timestamp with PSDs and appended Quality.
        Returns:
            Features, format according to the selected option.
        """
        verbose = 1
        # param
        add_quality = False
        return_by_hours = True

        # Extract features from the patient metadata.
        age = hp.get_age(patient_metadata)
        sex = hp.get_sex(patient_metadata)
        rosc = hp.get_rosc(patient_metadata)
        ohca = hp.get_ohca(patient_metadata)
        vfib = hp.get_vfib(patient_metadata)
        ttm = hp.get_ttm(patient_metadata)
        quality_scores = hp.get_quality_scores(recording_metadata)

        # Use one-hot encoding for sex; add more variables
        sex_features = np.zeros(2, dtype=int)
        if sex == 'Female':
            female = 1
            male   = 0
            other  = 0
        elif sex == 'Male':
            female = 0
            male   = 1
            other  = 0
        else:
            female = 0
            male   = 0
            other  = 1

        # Combine the patient features.
        patient_features = np.array([age, female, male, other, rosc, ohca, vfib, ttm])
        print(patient_features)
        # Extract features from the recording data and metadata.
        channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
                    'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
        num_channels = len(channels)
        num_recordings = len(recording_data)

        # 1. Compute mean and standard deviation for each channel for each recording.
        # 2. Compute the power spectral density for the delta, theta, alpha, and beta frequency bands 
        #    for each channel from all the recording.
        available_signal_data = list()
        delta_psd_data = list()
        theta_psd_data = list()
        alpha_psd_data = list()
        beta_psd_data  = list()
        # num recordings is always 72
        for i in range(num_recordings):
            signal_data, sampling_frequency, signal_channels = recording_data[i]
            if signal_data is not None:
                signal_data = hp.reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.
            
                delta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False)
                theta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
                alpha_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
                beta_psd,  _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)
                # print(delta_psd)
                # print(theta_psd)
                # print(alpha_psd)
                # print(beta_psd)
                quality_score = list()
                quality_score.append(quality_scores[i])

                if add_quality == True: # num_channels (+1)
                    signal_data = np.append(signal_data, [quality_score * signal_data.shape[1]], axis=0)
                    delta_psd   = np.append(delta_psd, [quality_score * delta_psd.shape[1]], axis=0)
                    theta_psd   = np.append(theta_psd, [quality_score * theta_psd.shape[1]], axis=0)
                    alpha_psd   = np.append(alpha_psd, [quality_score * alpha_psd.shape[1]], axis=0)
                    beta_psd    = np.append(beta_psd, [quality_score * beta_psd.shape[1]], axis=0)                

                # DEBUG
                # print("shapes:", signal_data.shape, delta_psd.shape, theta_psd.shape, alpha_psd.shape, beta_psd.shape)
                
                available_signal_data.append(signal_data)
                delta_psd_data.append(delta_psd)
                theta_psd_data.append(theta_psd)
                alpha_psd_data.append(alpha_psd)
                beta_psd_data.append(beta_psd)

        if len(available_signal_data) > 0:

            if return_by_hours == True:
                stack_func = np.dstack            
                trans_axes = (2, 1, 0)
            else:
                stack_func = np.hstack
                trans_axes = (1, 0)
            
            available_signal_data = stack_func(available_signal_data)
            delta_psd_data = stack_func(delta_psd_data)
            theta_psd_data = stack_func(theta_psd_data)
            alpha_psd_data = stack_func(alpha_psd_data)
            beta_psd_data = stack_func(beta_psd_data)

            available_signal_data = np.transpose(available_signal_data, trans_axes)
            delta_psd_data = np.transpose(delta_psd_data, trans_axes)
            theta_psd_data = np.transpose(theta_psd_data, trans_axes)
            alpha_psd_data = np.transpose(alpha_psd_data, trans_axes)
            beta_psd_data = np.transpose(beta_psd_data, trans_axes)

        else:
            print("am I here")
            # print(hp.get_patient_id(patient_metadata))
            available_signal_data = delta_psd_data = theta_psd_data = alpha_psd_data = beta_psd_data = np.hstack(float('nan') * np.ones(num_channels))
            available_signal_data = np.empty((1, 30000, 18))
            available_signal_data.fill(np.NaN)
        
        
        
        return patient_features, available_signal_data, delta_psd_data, theta_psd_data, alpha_psd_data, beta_psd_data
        # return available_signal_data




