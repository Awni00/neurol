'''
Module including utility functions for creating `classifier`'s,
`transfromer`'s, and `calibrator`'s for use in the `BCI` module.
'''

import time

try:
    import biosppy.signals as bsig
except ImportError:
    raise ImportError(
        "biosppy is not installed. \n"
        "biosppy is required for some functionality in BCI_tools.\n"
        "you can install it using `pip install biosppy`")

import numpy as np

from .models.classification_tools import get_channels
from .models.preprocessing import epoch, epoch_band_features

# is this right? is it 220 Hz (see documentation)?
DEVICE_SAMPLING_RATE = {'muse': 256,
                        }

# FIXME: go through and fix
# what to do w/ DEVICE_SAMPLING_RATE dict
# `transformers` not being used
# the use of `device`
def ensemble_transform(signal, epoch_len=None, channels=None, device=None,
                        transformers=None, filter_=False, sampling_rate=None,
                        filter_kwargs=None):
    '''
    Ensemble transform function. Takes in buffer as input. Extracts the
    appropriate channels and samples, performs filtering, and transforms.

    Arguments:
        signal(np.ndarray): signal of shape: [n_samples, n_channels]
        epoch_len(int): length of epoch expected by classifier (# of samples).
                        optional.
        channels(list of str or int): list of channels expected by classifier.
                        See get_channels. optional.
        device(str): device name. used to get channels and sampling_rate.
        filter_(boolean): whether to perform filtering
        filter_kwargs(dict): dictionary of kwargs passed to filtering function.
            See biosppy.signals.tools.filter_signal. by default,
            an order 8 bandpass butter filter is performed between 2Hz and 40Hz.
    '''

    transformed_signal = signal

    # get the latest epoch from the signal
    if epoch_len is not None:
        transformed_signal = np.array(signal[-1*epoch_len:, :])

    # get the selected channels
    if channels:
        transformed_signal = get_channels(transformed_signal, channels, device)

    # filter_signal
    if filter_:
        if not sampling_rate:
            try:
                sampling_rate = DEVICE_SAMPLING_RATE[device]
            except KeyError:
                raise ValueError(
                    'sampling_rate is required when filter_ is True.\n'
                    'Note: this can sometimes be extracted from '
                    'the `device` parameter for supported devices')
        else:
            if not filter_kwargs:
                filter_kwargs = {}
            transformed_signal = filter_signal(
                transformed_signal, sampling_rate, **filter_kwargs)


    # apply pipeline of transformers
    if transformers:
        for transformer in transformers:
            transformed_signal = transformer(transformed_signal)

    return transformed_signal


def filter_signal(signal, sampling_rate, ftype='butter', band='bandpass',
                  frequency=(2, 40), order=8, **filter_kwargs):
    """
    applies frequency-based filters to a given signal.

    Args:
        signal (np.ndarray): signal of shape [n_samples, n_channels]
        sampling_rate(float): sampling rate of signal.
        ftype (str, optional): type of filter.
            one of 'FIR', 'butter', 'chebby1', 'chebby2', 'ellip', or 'bessel'.
            Defaults to 'butter'.
        band (str, optional): band type.
            one of 'lowpass', 'highpass', 'bandpass', or 'bandstop'.
            Defaults to 'bandpass'.
        frequency (float or tuple of floats, optional): cutoff frequencies.
            single if 'lowpass'/'highpass', tuple if 'bandpass'/'bandstop'.
            Defaults to (2,40).
        order (int, optional): order of filter. Defaults to 8.
        **filter_kwargs: keyword args for biosppy.signals.tools.filter_signal

    Returns:
        [np.ndarray]: filtered signal
    """

    filtered, _, _ = bsig.tools.filter_signal(
        signal=signal.T, sampling_rate=sampling_rate, ftype=ftype, band=band,
        frequency=frequency, order=order, **filter_kwargs)
    filtered = filtered.T

    return filtered


def band_power_calibrator(stream, channels, device, bands, percentile=50,
                          recording_length=10, epoch_len=1,
                          inter_window_interval=0.2):
    '''
    Calibrator for `generic_BCI.BCI` which computes a given `percentile` for
    the power of each frequency band across epochs channel-wise. Useful for
    calibrating a concentration-based BCI.

    Arguments:
        stream(neurol.streams object): neurol stream for brain data.
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

    # sleep for recording_length while stream accumulates data
    # necessary so that no data is used before the indicated start of recording
    time.sleep(recording_length)
    recording = stream.get_data()  # get accumulated data
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

    return clb_info


def band_power_transformer(signal, sampling_rate, bands):
    '''
    Transformer for `generic_BCI.BCI` which chooses channels, epochs, and
    gets power features on some choice of bands.

    Arguments:
        signal(np.ndarray): most recent stream data.
            shape: [n_samples, n_channels]
        sampling_rate(float): sampling_rate of signal.
        bands: the frequency bands to get power features for.
            'all': all of ['theta', 'alpha_low', 'alpha_high', 'beta', 'gamma']
            otherwise a list of strings of the desired bands.

    Returns:
        transformed_signal: array of shape [n_bands, n_channels] of the
        channel-wise power of each band over the epoch.
    '''

    # compute band_features on signal
    transformed_signal = np.squeeze(epoch_band_features(
        signal, sampling_rate, bands=bands, return_dict=False))

    return transformed_signal
