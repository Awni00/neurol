'''
Module implementing a general Brain-Computer Interface which manages
and incoming stream of neural data and responds to it in real-time.
'''

import time

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
        calibration_info: the result of the calibrator, if applicable.
        buffer_length(int): the length of the `buffer`; specifies the
            number of samples of the signal to keep for classification.
        brain_state: the most recent brain state classification.

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

        self.brain_state = None

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

    def _update(self, buffer):

        if buffer.shape[0] > self.buffer_length:
            # clip to buffer_length
            buffer = buffer[-self.buffer_length:]

            # transform buffer for classification
            if self.transformer is not None:
                try:
                    clf_input = self.transformer(buffer, self.calibration_info)
                except TypeError as type_err:
                    print(('Got TypeError when calling transformer.\n\n'
                           'Make sure your transformer is a function '
                           'which accepts (buffer, calibration_info) as inputs.'
                           ))
                    raise type_err
            else:
                clf_input = buffer

            try:
                # perform classification
                self.brain_state = self.classifier(clf_input,
                                                   self.calibration_info)
            except TypeError as type_err:
                print(('Got TypeError when calling classifier. \n\n'
                       'Make sure your classifier is a function which '
                       'accepts (clf_input, calibration_info) as inputs.'))
                raise type_err

            try:
                # run action based on classification
                self.action(self.brain_state)
            except TypeError as type_err:
                print(('Got TypeError when calling action. \n\n'
                       'Make sure your action is a function which '
                       'accepts (brain_state) as an input.'))
                raise type_err

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

                self._update(buffer)

    def test_update_rate(self, inlet, test_length=10, perform_action=True):
        '''
        Returns the rate at which the BCI is able to make a classification
        and perform its action.

        Arguments:
            inlet: a pylsl `StreamInlet` of the brain signal.
            test_length(float): how long to run the test for in seconds.
            perform_action(bool): whether to perform the action or skip it.
        '''

        inlet.open_stream()

        # get number of available channels in inlet
        n_channels = inlet.channel_count
        buffer = np.empty((0, n_channels))  # initialize buffer

        n_updates = 0
        start_time = time.time()

        while time.time() - start_time < test_length:
            chunk, _ = inlet.pull_chunk(max_samples=self.buffer_length)
            if np.size(chunk) != 0:  # Check if new data available
                buffer = np.append(buffer, np.array(chunk), axis=0)

                self._update(buffer)

                n_updates += 1

        update_rate = n_updates / test_length

        return update_rate


class fsm_BCI(generic_BCI):
    '''
    Implements a Finite-State-Machine-inspired Brain-Computer Interface.

    Classification of brain-state is not only dependent on the transformed
    real-time brain signal, but also the previous brain state.

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
        calibration_info: the result of the calibrator, if applicable.
        buffer_length(int): the length of the `buffer`; specifies the
            number of samples of the signal to keep for classification.
        brain_state: the most recent brain state classification.

    '''

    def _update(self, buffer):
        if buffer.shape[0] > self.buffer_length:
            # clip to buffer_length
            buffer = buffer[-self.buffer_length:]

            # transform buffer for classification
            if self.transformer is not None:
                try:
                    clf_input = self.transformer(buffer, self.calibration_info)
                except TypeError as type_err:
                    print(('Got TypeError when calling transformer.\n\n'
                           'Make sure your transformer is a function '
                           'which accepts (buffer, calibration_info) as inputs.'
                           ))
                    raise type_err
            else:
                clf_input = buffer

            try:
                # perform classification
                self.brain_state = self.classifier(clf_input, self.brain_state,
                                                   self.calibration_info)
            except TypeError as type_err:
                print(('Got TypeError when calling classifier. \n\n'
                       'Make sure your classifier is a function which accepts '
                       '(clf_input, brain_state, calibration_info) as inputs.'))
                raise type_err

            try:
                # run action based on classification
                self.action(self.brain_state)
            except TypeError as type_err:
                print(('Got TypeError when calling action. \n\n'
                       'Make sure your action is a function which '
                       'accepts (brain_state) as an input.'))
                raise type_err


class retentive_BCI(generic_BCI):
    '''
    Implements a Brain-Computer Interface with memory of past brain states.

    Classification of brain-state is not only dependent on the transformed
    real-time brain signal, but also the finite list of previous brain states.

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
        calibration_info: the result of the calibrator, if applicable.
        buffer_length(int): the length of the `buffer`; specifies the
            number of samples of the signal to keep for classification.
        brain_state: the most recent brain state classification.
        memory_length(int): number of brain states into the past to remember.
        past_states: a list of the past classifications of brain states.
            used in next classification. length is memory_length.
    '''

    def __init__(self, classifier, transformer=None, action=print,
                 calibrator=None, buffer_length=1024, memory_length=10):
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
            memory_length(int): number of brain states to remember into past.
        '''

        super().__init__(classifier,
                         transformer, action, calibrator, buffer_length)

        self.memory_length = memory_length
        self.past_states = []

    def _update(self, buffer):
        if buffer.shape[0] > self.buffer_length:
            # clip to buffer_length
            buffer = buffer[-self.buffer_length:]

            # transform buffer for classification
            if self.transformer is not None:
                try:
                    clf_input = self.transformer(buffer, self.calibration_info)
                except TypeError as type_err:
                    print(('Got TypeError when calling transformer.\n\n'
                           'Make sure your transformer is a function '
                           'which accepts (buffer, calibration_info) as inputs.'
                           ))
                    raise type_err
            else:
                clf_input = buffer

            try:
                # perform classification
                self.brain_state = self.classifier(clf_input, self.past_states,
                                                   self.calibration_info)
                self.past_states.append(self.brain_state)
                if len(self.past_states) > self.memory_length:
                    self.past_states = self.past_states[-self.memory_length:]
            except TypeError as type_err:
                print(('Got TypeError when calling classifier. \n\n'
                       'Make sure your classifier is a function which accepts '
                       '(clf_input, past_states, calibration_info) as inputs.'))
                raise type_err

            try:
                # run action based on classification
                self.action(self.brain_state)
            except TypeError as type_err:
                print(('Got TypeError when calling action. \n\n'
                       'Make sure your action is a function which '
                       'accepts (brain_state) as an input.'))
                raise type_err
