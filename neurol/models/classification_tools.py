'''
Module containing functions for performing classification, via machine learning
models or otherwise, related to Brain-Computer Interface applications.
'''

import numpy as np

# is this right? is it 220 Hz (see documentation)?
DEVICE_SAMPLING_RATE = {'muse': 256,
                        }


def get_channels(signal, channels, device='muse'):
    '''
    Returns a signal with only the desired channels.

    Arguments:
        signal: a signal of shape [n_samples, n_channels]
        channels: an array of the str names of the desired channels.
            returned in given order.
        device: str name of the device.

    Returns:
        numpy array of signal with shape [n_channels, n_desired_channels].
        Includes only the selected channels in the order given.
    '''

    # check device; each device has its own ch_ind dictionary corresponding to
    # its available channels
    if device == 'muse':
        ch_ind_muse = {'TP9': 0, 'AF7': 1, 'AF8': 2, 'TP10': 3}
        return_signal = np.array([signal[:, ch_ind_muse[ch]]
                                  for ch in channels]).T

    return return_signal


def softmax_predict(input_, predictor, thresh=0.5):
    '''
    Consolidates a softmax prediction to a one-hot encoded prediction.

    Arguments:
        input_: the input taken by the predictor
        predictor: function which returns a softmax prediction given an input_
        thresh: the threshold for a positive prediction for a particular class.

    '''

    pred = np.array(predictor(input_))

    return (pred >= thresh).astype(int)


def encode_ohe_prediction(prediction):
    '''
    Returns the index number of the positive class in a
    one-hot encoded prediction.
    '''
    return np.where(np.array(prediction) == 1)[0][0]


def decode_prediction(prediction, decode_dict):
    '''
    Returns a more intelligible reading of the prediction
    based on the given decode_dict
    '''
    return decode_dict[prediction]
