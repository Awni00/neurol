'''
Module implementing a general Brain-Computer Interface which manages
and incoming stream of neural data and responds to it in real-time.
'''

import numpy as np


class generic_BCI:
    '''
    Implements a generic Brain-Computer Interface.

    Internally manages a buffer of the signal given in `inlet` and
    continuously performs classification on appropriately transformed
    real-time neural data. At each classification, a corresponding `action`
    is performed.

    Attributes:
        classifier: a function which performs classification on the
            most recent data (transformed as needed). returns classification.
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

    def __init__(self, classifier, transformer=None, action=print,
                 calibrator=None, buffer_length=1024):
        '''
        Initialize a generic BCI object.

        See class documentation for infromation about the class itself.

        Arguments:
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

        self.classifier = classifier
        self.transformer = transformer
        self.action = action

        self.calibrator = calibrator
        self.calibration_info = None

        self.buffer_length = buffer_length

    def calibrate(self, inlet):
        '''
        runs the `calibrator`.

        return value of `calibrator` is stored in the instance's
        `calibration_info` which the `transformer` and `classifier`
        can use at run-time of BCI.

        Arguments:
            inlet: a pylsl `StreamInlet` of the brain signal.
        '''
        # run calibrator to get `calibration_info` which is used when
        # perfroming transformation and classifiaction
        if self.calibrator is not None:
            self.calibration_info = self.calibrator(inlet)
        else:
            print('Instance of generic BCI has no calibrator')
            self.calibration_info = None

    def run(self, inlet):
        '''
        Runs the defined Brain-Computer Interface.

        Internally manages a buffer of the signal given in `inlet` and
        continuously performs classification on appropriately transformed
        real-time neural data. At each classification, a corresponding `action`
        is performed.


        Arguments:
            inlet: a pylsl `StreamInlet` of the brain signal.
        '''

        inlet.open_stream()

        # get number of available channels in inlet
        n_channels = inlet.channel_count
        buffer = np.empty((0, n_channels))  # initialize buffer

        # TODO: implement ending condition?
        running = True  # currently constantly running.

        while running:
            chunk, _ = inlet.pull_chunk(max_samples=self.buffer_length)
            if np.size(chunk) != 0:  # Check if new data available
                buffer = np.append(buffer, np.array(chunk), axis=0)

                if buffer.shape[0] > self.buffer_length:
                    # clip to buffer_length
                    buffer = buffer[-self.buffer_length:]

                    # transform buffer for classification
                    if self.transformer is not None:
                        try:
                            clf_input = self.transformer(
                                buffer, self.calibration_info)
                        except TypeError as type_err:
                            # NOTE: should this be formatted differently?
                            print(('Got TypeError when calling transformer.\n'
                                   'Make sure your transformer is a function'
                                   'which accepts buffer, and calibration_info'
                                   '(output of calibrator) as inputs'))
                            print(type_err)
                            break
                    else:
                        clf_input = buffer

                    try:
                        # perform classification
                        brain_state = self.classifier(
                            clf_input, self.calibration_info)
                    except TypeError as type_err:
                        # NOTE: should this be formatted differently?
                        print(('Got TypeError when calling classifier. \n'
                               'Make sure your classifier is a function which '
                               'accepts clf_input (output of transformer), and '
                               'calibration_info (output of calibrator) as '
                               'inputs'))
                        print(type_err)
                        break

                    try:
                        # run action based on classification
                        self.action(brain_state)
                    except TypeError as type_err:
                        print(('Got TypeError when calling action. \n'
                               'Make sure your action is a function which '
                               'accepts brain_state (output of classifer) '
                               'as an input'))
                        print(type_err)
                        break

        inlet.close_stream()
