'''
Module including utility functions for creating `classifier`'s,
`transfromer`'s, and `calibrator`'s for use in the `BCI` module.
'''

import time

import biosppy.signals as bsig
import numpy as np

from .models.classification_tools import DEVICE_SAMPLING_RATE, get_channels
from .models.preprocessing import epoch, epoch_band_features


def ensemble_transform(buffer, epoch_len, channels, device='muse',
                       transformers=None, filter_=False, filter_kwargs=None):
    '''
    Ensemble transform function. Takes in buffer as input. Extracts the
    appropriate channels and samples, performs filtering, and transforms.

    Arguments:
        buffer: most recent stream data. shape: [n_samples, n_channels]
        epoch_len: length of epoch expected by predictor (number of samples).
        channels: list of channels expected by predictor. See get_channels.
        device: string of device name. used to get channels and sampling_rate.
        filter_: boolean of whether to perform filtering
        filter_kwargs: dictionary of kwargs to be passed to filtering function.
            See biosppy.signals.tools.filter_signal. by default,
            an order 8 bandpass butter filter is performed between 2Hz and 40Hz.
    '''

    # get the latest epoch_len samples of the buffer
    transformed_signal = np.array(buffer[-epoch_len:, :])

    # get the selected channels
    transformed_signal = get_channels(transformed_signal, channels, device)

    # filter_signal
    if filter_:
        # create dictionary of kwargs for filter_signal
        filt_kwargs = {'sampling_rate': DEVICE_SAMPLING_RATE[device],
                       'ftype': 'butter',
                       'band': 'bandpass',
                       'frequency': (2, 40),
                       'order': 8}

        if filter_kwargs is not None:
            filt_kwargs.update(filter_kwargs)

        transformed_signal, _, _ = bsig.tools.filter_signal(
            signal=transformed_signal.T, **filt_kwargs)
        transformed_signal = transformed_signal.T

    return transformed_signal


def band_power_calibrator(inlet, channels, device, bands, percentile=50,
                          recording_length=10, epoch_len=1,
                          inter_window_interval=0.2):
    '''
    Calibrator for `generic_BCI.BCI` which computes a given `percentile` for
    the power of each frequency band across epochs channel-wise. Useful for
    calibrating a concentration-based BCI.

    Arguments:
        inlet: a pylsl `StreamInlet` of the brain signal.
        channels: array of strings with the names of channels to use.
        device(str): device name for use by `classification_tools`
        bands: the frequency bands to get power features for.
            'all': all of ['theta', 'alpha_low', 'alpha_high', 'beta', 'gamma']
            otherwise an array of strings of the desired bands.
        percentile: the percentile of power distribution across epochs to
            return for each band.
        recording_length(float): length of recording to use for calibration
            in seconds.
        epoch_len(float): the length of each epoch in seconds.
        inter_window_interval(float): interval between each window/epoch
            in seconds (measured start to start)

    Returns:
        clb_info: array of shape [n_bands, n_channels] of the `percentile`
        of the power of each band
    '''
    sampling_rate = DEVICE_SAMPLING_RATE[device]  # get sampling_rate
    # calculate window size in # of samples
    ws = int(epoch_len * sampling_rate)
    # calculate inter_window_interval in # of samples
    iwi = int(inter_window_interval * sampling_rate)

    input('Press Enter to begin calibration...')

    print(f'Recording for {recording_length} seconds...')

    # sleep for recording_length while inlet accumulates chunk
    # necessary so that no data is used before the indicated start of recording
    time.sleep(recording_length)
    recording, _ = inlet.pull_chunk(
        max_samples=sampling_rate*recording_length)  # get accumulated data
    # get appropriate channels
    recording = get_channels(np.array(recording), channels, device=device)

    # epoch the recording to compute percentile across distribution
    epochs = epoch(recording, ws, iwi)

    # compute band power for each epoch
    band_power = np.array([epoch_band_features(epoch, sampling_rate,
                                               bands=bands, return_dict=False)
                           for epoch in epochs])

    # calculate given percentile of band power
    clb_info = np.squeeze(np.percentile(band_power, percentile, axis=0))

    print(f'\nComputed the following power percentiles: \n{clb_info}')
    input('\nCalibration complete. Press Enter to start BCI...')

    return clb_info


def band_power_transformer(buffer, clb_info, channels, device, bands,
                           epoch_len=1):
    '''
    Transformer for `generic_BCI.BCI` which chooses channels, epochs, and
    gets power features on some choice of bands.

    Arguments:
        buffer: most recent stream data. shape: [n_samples, n_channels]
        clb_info: not used. included for compatibility with generic_BCI.BCI
        channels: list of strings of the channels to use.
        device:(str): device name for use by `classification_tools`.
        bands: the frequency bands to get power features for.
            'all': all of ['theta', 'alpha_low', 'alpha_high', 'beta', 'gamma']
            otherwise a list of strings of the desired bands.
        epoch_len(float): the duration of data to classify on in seconds.

    Returns:
        transformed_signal: array of shape [n_bands, n_channels] of the
        channel-wise power of each band over the epoch.
    '''
    sr = DEVICE_SAMPLING_RATE[device]  # get device sampling rate

    # get the latest epoch_len samples from the buffer
    transformed_signal = np.array(buffer[-int(epoch_len*sr):, :])

    # get the selected channels
    transformed_signal = get_channels(transformed_signal, channels, device)

    # compute band_features on signal
    transformed_signal = np.squeeze(epoch_band_features(
        transformed_signal, sr, bands=bands, return_dict=False))

    return transformed_signal
