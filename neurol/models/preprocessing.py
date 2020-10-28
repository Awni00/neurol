'''
Module containing functions for the preparation of neural data for use with
with BCI-related models.

'''

import numpy as np
from scipy import stats

try:
    import biosppy.signals as bsig
except ImportError:
    raise ImportError(
        "biosppy is not installed. \n"
        "biosppy is required for some preprocessing functionality.\n"
        "you can install it using `pip install biosppy`")


def epoch(signal, window_size, inter_window_interval):
    '''
    Creates overlapping windows/epochs of EEG data from a single recording.

    Arguments:
        signal: array of timeseries EEG data of shape [n_samples, n_channels]
        window_size(int): desired size of each window in number of samples
        inter_window_interval(int): interval between each window in number of
            samples (measured start to start)

    Returns:
        numpy array object with the epochs along its first dimension
    '''

    # Calculate the number of valid windows of the specified size in signal
    num_windows = 1 + (len(signal) - window_size) // inter_window_interval

    epochs = []
    for i in range(num_windows):
        epochs.append(signal[i*inter_window_interval:i *
                             inter_window_interval + window_size])

    return np.array(epochs)


def labels_from_timestamps(timestamps, sampling_rate, length):
    '''takes an array containing timestamps (as floats) and
    returns a labels array of size 'length' where each index
    corresponding to a timestamp via the 'samplingRate'.

    Arguments:
        timestamps: an array of floats containing the timestamps for each
        event (units matching sampling_rate).
        sampling_rate(float): the sampling rate of the EEG data.
        length(int): the number of samples of the corresponing EEG recording.

    Returns:
        an integer array of size 'length' with a '1' at each time index where a
        corresponding timestamp exists, and a '0' otherwise.
    '''
    # NOTE:  currently assumes binary labels. Generalize to arbitrary labels.

    # create labels array template
    labels = np.zeros(length)

    # calculate corresponding index for each timestamp
    labelIndices = np.array(timestamps * sampling_rate)
    labelIndices = labelIndices.astype(int)

    # flag label at each index where a corresponding timestamp exists
    labels[labelIndices] = 1
    labels = labels.astype(int)

    return np.array(labels)


def label_epochs(labels, window_size, inter_window_interval, label_method):
    '''
    create labels for individual eoicgs of EEG data based on the
    label_method.

    Arguments:
        labels: an integer array indicating a class for each sample measurement
        window_size(int): size of each window in number of samples
            (matching window_size in epoched data)
        inter_window_interval(int): interval between each window in number of
            samples (matching inter_window_interval in epoched data)
        label_method(str/func): method of consolidating labels contained in
            epoch into a single label.
                'containment': whether a positive label occurs in the epoch,
                'count': the count of positive labels in the epoch,
                'mode': the most common label in the epoch
                func: func_name(epoched_labels) outputs label of epoched_labels

    Returns:
        a numpy array with a label correponding to each epoch
    '''

    # epoch the labels themselves so each epoch contains a label at each sample
    epochs = epoch(labels, window_size, inter_window_interval)

    # if a positive label [1] occurs in the epoch, give epoch positive label
    if label_method == 'containment':
        epoch_labels = [int(1 in epoch) for epoch in epochs]

    elif label_method == 'count':
        # counts the number of occurences of the positive label [1]]
        epoch_labels = [epoch.count(1) for epoch in epochs]

    elif label_method == 'mode':
        # choose the most common label occurence in the epoch
        # default to the smallest if multiple exist
        epoch_labels = [stats.mode(epoch)[0][0] for epoch in epochs]

    elif callable(label_method):
        epoch_labels = [label_method(epoch) for epoch in epochs]
    else:
        raise TypeError("label_method is invalid.")

    return np.array(epoch_labels)


def label_epochs_from_timestamps(timestamps, sampling_rate, length,
                                 window_size, inter_window_interval,
                                 label_method='containment'):
    '''
    Directly creates labels for individual windows of EEG data from
    timestamps of events.

    Arguments:
        timestamps: an array of floats containing the timestamps
            for each event (units matching sampling_rate).
        sampling_rate(float): sampling rate of the recording.
        length(int): the number of samples of the corresponing EEG recording.
        window_size(int): size of each window in number of samples
            (matches window_size in epoched data)
        inter_window_interval(int): interval between each window in number of
            samples (matches inter_window_interval in epoched data)
        label_method(str/func): method of consolidating labels contained in
            epoch into a single label.
                'containment': whether a positive label occurs in the epoch,
                'count': the count of positive labels in the epoch,
                'mode': the most common label in the epoch
                func: func_name(epoched_labels) outputs label of epoched_labels

    Returns:
        an array with a label correponding to each window
    '''
    labels = labels_from_timestamps(timestamps, sampling_rate, length)

    epoch_labels = label_epochs(
        labels, window_size, inter_window_interval, label_method)

    return epoch_labels


def epoch_and_label(data, sampling_rate, timestamps, window_size,
                    inter_window_interval, label_method='containment'):
    '''
    Epochs a signal (single EEG recording) and labels each epoch using
    timestamps of events and a chosen labelling method.

    Arguments:
        data: array of timeseries EEG data of shape [n_samples, n_channels]
        timestamps: an array of floats containing the timestamps for each event
            in units of time.
        sampling_rate(float): the sampling rate of the EEG data.
        window_size(float): desired size of each window in units of time.
        inter_window_interval(float): interval between each window
            in units of time (measured start to start)
        label_method(str/func): method of consolidating labels contained in
            epoch into a single label.
                'containment': whether a positive label occurs in the epoch,
                'count': the count of positive labels in the epoch,
                'mode': the most common label in the epoch
                func: func_name(epoched_labels) outputs label of epoched_labels

    Returns:
        epochs: array of epochs with shape [n_epochs, n_channels]
        labels: array of labels corresponding to each epoch of shape [n_epochs,]
    '''

    ws = int(window_size*sampling_rate)
    iwi = int(inter_window_interval*sampling_rate)
    epochs = epoch(data, ws, iwi)
    labels = label_epochs_from_timestamps(timestamps, sampling_rate, len(data),
                                          ws, iwi, label_method=label_method)

    return epochs, labels


def compute_signal_std(signal, corrupt_intervals=None, sampling_rate=1):
    '''
    Computes and returns the standard deviation of a signal channel-wise
    while avoiding corrupt intervals

    Arguments:
        signal: signal of shape [n_samples, n_channels]
        corrupt_intervals: an array of 2-tuples indicating the start and
            end time of the corrupt interval (units of time)
        sampling_rate: the sampling rate in units of samples/unit of time


    Returns:
        standard deviation of signal channel-wise of shape [1, n_channels]
    '''

    # convert signal into numpy array for use of its indexing features
    signal = np.array(signal)

    if corrupt_intervals is not None:
        # convert corrupt_indices from units of time to # of samples
        corrupt_intervals = [(int(cor_start * sampling_rate),
                              int(cor_end * sampling_rate))
                             for cor_start, cor_end in corrupt_intervals]

        # find all indices to keep
        good_indices = []

        # for each index in the signal,
        # check if it is contained in any of the corrupt intervals
        for ind in range(len(signal)):
            good_indices.append(not np.any(
                [ind in range(cor_start, cor_end)
                 for cor_start, cor_end in corrupt_intervals]))

        signal = signal[good_indices]

    # compute and return std on non_corrupt parts of signal
    return np.std(signal, axis=0, dtype=np.float32)


def split_corrupt_signal(signal, corrupt_intervals, sampling_rate=1):
    '''
    Splits a signal with corrupt intervals and returns array of signals
    with the corrupt intervals filtered out. This is useful for treating
    each non_corrupt segment as a seperate signal to ensure continuity
    within a single signal.

    Arguments:
        signal: signal of shape [n_samples, n_channels]
        corrupt_intervals: an array of 2-tuples indicating the start and
            end time of the corrupt interval (units of time)
        sampling_rate: the sampling rate in units of samples/unit of time


    Returns:
        array of non_corrupt signals of shape [n_signal, n_samples, n_channels]
    '''

    # convert corrupt_indices from units of time and flatten into single array
    corrupt_intervals = np.array(corrupt_intervals * sampling_rate).flatten()

    # convert signal into numpy array for use with library
    signal = np.array(signal)

    # split signal on each cor_start/cor_end and discard the regions in between
    # cor_start and cor_end
    signals = np.split(signal, corrupt_intervals)[::2]

    return signals


def epoch_band_features(epoch_, sampling_rate, bands='all', return_dict=True):
    '''
    Computes power features of EEG frequency bands over the epoch channel-wise.

    Arguments:
        epoch_: a single epoch of shape [n_samples, n_channels]
        sampling_rate: the sampling rate of the signal in units of samples/sec
        bands: the requested frequency bands to get power features for.
            'all': all of ['theta', 'alpha_low', 'alpha_high', 'beta', 'gamma']
            otherwise an array of strings of the desired bands.
        return_dict(bool): returns `band_features` in the form of a dictionary
            if True, else returns as numpy array in order of `bands`

    Returns:
        a dictionary of arrays of shape [1, n_channels] containing the
        power features over each frequency band per channel.
    '''

    if bands == 'all':
        bands = ['theta', 'alpha_low', 'alpha_high', 'beta', 'gamma']

    # computes length of epoch to compute band power features over entire epoch
    w_size = np.shape(epoch_)[0] / sampling_rate

    # compute power for each band
    _, theta, alpha_low, alpha_high, beta, gamma = bsig.eeg.get_power_features(
        epoch_, sampling_rate=sampling_rate, size=w_size, overlap=0)

    # initialize return dict
    band_features = {'theta': theta,
                     'alpha_low': alpha_low,
                     'alpha_high': alpha_high,
                     'beta': beta,
                     'gamma': gamma}

    # return only requested bands
    # (intersection of bands available and bands requested)
    if return_dict:  # return in form of dict
        return_band_features = {
            band: band_features[band] for band in bands & band_features.keys()}
    else:
        return_band_features = np.array(
            [band_features[band] for band in bands & band_features.keys()])

    return return_band_features
