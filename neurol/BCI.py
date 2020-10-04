'''
Module implementing a general Brain-Computer Interface which manages
and incoming stream of neural data and responds to it in real-time.
'''

import time


class generic_BCI:
    '''
    Implements a generic Brain-Computer Interface.

    Internally manages a buffer of the signal given in `stream` and
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
            calibration using `stream`; returns `calibration_info`
            which is used by `classifier` and `transformer`.
        calibration_info: the result of the calibrator, if applicable.
        buffer_length(int): the length of the `buffer`; specifies the
            number of samples of the signal to keep for classification.
        brain_state: the most recent brain state classification.

    '''

    def __init__(self, classifier, transformer=None, action=print,
                 calibrator=None):
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
                calibration using `stream`; returns `calibration_info`
                which is used by `classifier` and `transformer`.
        '''

        self.brain_state = None

        self.classifier = classifier
        self.transformer = transformer
        self.action = action

        self.calibrator = calibrator
        self.calibration_info = None

    def calibrate(self, stream):
        '''
        runs the `calibrator`.

        return value of `calibrator` is stored in the object's
        `calibration_info` which the `transformer` and `classifier`
        can use at run-time of BCI.

        Arguments:
            stream(neurol.streams object): neurol stream for brain data.
        '''
        # run calibrator to get `calibration_info` which is used when
        # perfroming transformation and classifiaction
        if self.calibrator is not None:
            self.calibration_info = self.calibrator(stream)
        else:
            print('Instance of BCI has no calibrator')
            self.calibration_info = None

    def _update(self, buffer):
        '''transforms, classifies, and acts on data in buffer'''

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

        # perform classification
        try:
            self.brain_state = self.classifier(clf_input,
                                               self.calibration_info)
        except TypeError as type_err:
            print(('Got TypeError when calling classifier. \n\n'
                   'Make sure your classifier is a function which '
                   'accepts (clf_input, calibration_info) as inputs.'))
            raise type_err

        # run action based on classification
        try:
            self.action(self.brain_state)
        except TypeError as type_err:
            print(('Got TypeError when calling action. \n\n'
                   'Make sure your action is a function which '
                   'accepts (brain_state) as an input.'))
            raise type_err

    def run(self, stream):
        '''
        Runs the defined Brain-Computer Interface.

        Internally manages a buffer of the signal given in `stream` and
        continuously performs classification on appropriately transformed
        real-time neural data. At each classification, a corresponding `action`
        is performed.


        Arguments:
            stream(neurol.streams object): neurol stream for brain data.
        '''

        # TODO: implement ending condition?
        running = True  # currently constantly running.

        while running:
            # if new data available, run _update on it
            if stream.update_buffer():
                self._update(stream.buffer)

    def test_update_rate(self, stream, test_length=10, perform_action=True):
        '''
        Returns the rate at which the BCI is able to make a classification
        and perform its action.

        Arguments:
            stream(neurol.streams object): neurol stream for brain data.
            test_length(float): how long to run the test for in seconds.
            perform_action(bool): whether to perform the action or skip it.
        '''

        n_updates = 0
        start_time = time.time()

        while time.time() - start_time < test_length:

            # if new data available, run _update on it
            if stream.update_buffer():
                self._update(stream.buffer)

                n_updates += 1

        update_rate = n_updates / test_length

        return update_rate


class fsm_BCI(generic_BCI):
    '''
    Implements a Finite-State-Machine-inspired Brain-Computer Interface.

    Classification of brain-state is not only dependent on the transformed
    real-time brain signal, but also the previous brain state.

    Internally manages a buffer of the signal given in `stream` and
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
            calibration using `stream`; returns `calibration_info`
            which is used by `classifier` and `transformer`.
        calibration_info: the result of the calibrator, if applicable.
        buffer_length(int): the length of the `buffer`; specifies the
            number of samples of the signal to keep for classification.
        brain_state: the most recent brain state classification.

    '''

    def _update(self, buffer):
        '''transforms, classifies, and acts on data in buffer'''

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

    Internally manages a buffer of the signal given in `stream` and
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
            calibration using `stream`; returns `calibration_info`
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
                calibration using `stream`; returns `calibration_info`
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
        '''transforms, classifies, and acts on data in buffer'''

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
