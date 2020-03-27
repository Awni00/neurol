'''
Module implementing a general Brain-Computer Interface which manages
and incoming stream of neural data and responds to it in real-time.
'''

import numpy as np


def generic_BCI(inlet, classifier, transformer=None, action=print,
                calibrator=None, buffer_length=1024):
    '''
    Implements a generic Brain-Computer Interface.

    Internally manages a buffer of the signal given in `inlet` and
    continuously performs classification on appropriately transformed
    real-time neural data. At each classification, a corresponding `action`
    is performed.


    Arguments:
        inlet: a pylsl `StreamInlet` of the brain signal.
        classifier: a function which performs classification on the
            most recent data (transformed as needed). returns class.
        transformer: function which takes in the most recent data (`buffer`)
            and returns the transformed input the classifer expects.
        action: a function which takes in the classification, and
            performs some action.
        calibrator: a function which is run on startup to perform
            calibration using `inlet`; returns `calibration_info`
            which is used by `classifier` and `transformer`.
        buffer_length(int): the length of the `buffer`; specifies the
            number of samples of the signal to keep for classification.
    '''

    inlet.open_stream()

    # run calibrator to get `calibration_info` which is used when
    # perfroming transformation and classifiaction
    if calibrator is not None:
        calibration_info = calibrator(inlet)
    else:
        calibration_info = None

    # get number of available channels in inlet
    n_channels = inlet.channel_count
    buffer = np.empty((0, n_channels)) #initialize buffer

    #TODO: implement ending condition?
    running = True  # currently constantly running.

    while running:
        chunk, _ = inlet.pull_chunk(max_samples=buffer_length)
        if np.size(chunk) != 0:  # Check if new data available
            buffer = np.append(buffer, np.array(chunk), axis=0)

            if buffer.shape[0] > buffer_length:
                buffer = buffer[-buffer_length:]  # clip to buffer_length

                # transform buffer for classification
                if transformer is not None:
                    try:
                        clf_input = transformer(buffer, calibration_info)
                    except TypeError as type_err:
                        print(('Got TypeError when calling transformer.\n'
                               'Make sure your transformer is a function \n'
                               'which accepts buffer, and calibration_info \n'
                               '(output of calibrator) as inputs'))
                        print(type_err)
                        break
                else:
                    clf_input = buffer

                try:
                    # perform classification
                    brain_state = classifier(clf_input, calibration_info)
                except TypeError as type_err:
                    print(('Got TypeError when calling classifier. \n'
                           'Make sure your classifier is a function which\n'
                           'accepts clf_input (output of transformer), and \n'
                           'calibration_info (output of calibrator) as inputs'))
                    print(type_err)
                    break

                try:
                    action(brain_state)  # run action based on classification
                except TypeError as type_err:
                    print(('Got TypeError when calling action. \n'
                           'Make sure your action is a function which accepts\n'
                           'brain_state (output of classifer) as an input'))
                    print(type_err)
                    break

    inlet.close_stream()
