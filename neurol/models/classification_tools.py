'''
Module containing functions for performing classification, via machine learning
models or otherwise, related to Brain-Computer Interface applications.
'''

import numpy as np



def get_channels(signal, channels, device=None):
    '''
    Returns a signal with only the desired channels.

    Arguments:
        signal(np.ndarray): a signal of shape [n_samples, n_channels]
        channels(array): str names or int indices of the desired channels.
            returned in given order.
        device(str): name of the device. Optional.

    Returns:
        numpy array of signal with shape [n_channels, n_desired_channels].
        Includes only the selected channels in the order given.
    '''

    # check device; each device has its own ch_ind dictionary corresponding to
    # its available channels
    if isinstance(channels[0], str):
        if device == 'muse':
            ch_ind_muse = {'TP9': 0, 'AF7': 1, 'AF8': 2, 'TP10': 3}
            return_signal = np.array([np.array(signal)[:, ch_ind_muse[ch]]
                                    for ch in channels]).T
        else:
            raise ValueError(
                'Given device is not supported. '
                'You should extract the desired channels manually.')

    elif isinstance(channels[0], int):
        return_signal = np.array([np.array(signal)[:, ch]
                                    for ch in channels]).T

    else:
        raise ValueError('Invalid channel type. Must be str name or int index.')

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


def threshold_clf(features, threshold, clf_consolidator='any'):
    '''
      Classifies given features based on a given threshold.

      Arguments:
          features: an array of numerical features to classify
          threshold: threshold for classification. A single number, or an
            array corresponding to `features` for element-wise comparison.
          clf_consolidator: method of consolidating element-wise comparisons
            with threshold into a single classification.
              'any': positive class if any features passes the threshold
              'all': positive class only if all features pass threshold
              'sum': a count of the number of features which pass the threshold
              function: a custom function which takes in an array of booleans
                and returns a consolidated classification

      Returns:
          classification for the given features. Return type `clf_consolidator`.
    '''

    try:
        label = np.array(features) > np.array(threshold)
    except ValueError as v_err:
        print("Couldn't perform comparison between features and thresholds."
              "Try a different format for the threshold.")
        raise v_err

    # consolidate binary label array into single classification
    if clf_consolidator == 'any':
        label = np.any(label)

    elif clf_consolidator == 'all':
        label = np.all(label)

    elif clf_consolidator == 'sum':
        label = np.sum(label)

    elif callable(clf_consolidator):
        try:
            label = clf_consolidator(label)
        except TypeError as t_err:
            print("Couldn't consolidate classification with `clf_consolidator`")
            raise t_err
    else:
        raise ValueError('The given clf_consolidator is not supported.')

    return label
