'''
Module implementing a general Brain-Computer Interface which manages
and incoming stream of neural data and responds to it in real-time.
'''

import time
import numpy as np


class generic_BCI:
    '''
    Implements a generic Brain-Computer Interface.

    Internally manages a buffer of the signal given in `stream` and
    continuously performs classification on appropriately transformed
    real-time neural data. At each classification, a corresponding `action`
    is performed.

    Attributes:
        classifier (function): a function which performs classification on the
            most recent data (transformed as needed). returns classification.
        transformer (function): function which takes in the most recent data (`buffer`)
            and returns the transformed input the classifer expects.
        action (function): a function which takes in the classification, and
            performs some action.
        calibrator (function): a function which is run on startup to perform
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
            classifier (function): a function which performs classification on the
                most recent data (transformed as needed). returns class.
            transformer (function): function which takes in the most recent data (`buffer`)
                and returns the transformed input the classifer expects.
            action (function): a function which takes in the classification, and
                performs some action.
            calibrator (function): a function which is run on startup to perform
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
            stream (neurol.streams object): neurol stream for brain data.
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
            stream (neurol.streams object): neurol stream for brain data.
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
            stream (neurol.streams object): neurol stream for brain data.
            test_length (float): how long to run the test for in seconds.
            perform_action (bool): whether to perform the action or skip it.
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
        classifier (function): a function which performs classification on the
            most recent data (transformed as needed). returns classification.
        transformer (function): function which takes in the most recent data (`buffer`)
            and returns the transformed input the classifer expects.
        action (function): a function which takes in the classification, and
            performs some action.
        calibrator (function): a function which is run on startup to perform
            calibration using `stream`; returns `calibration_info`
            which is used by `classifier` and `transformer`.
        calibration_info: the result of the calibrator, if applicable.
        buffer_length (int): the length of the `buffer`; specifies the
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
        classifier (function): a function which performs classification on the
            most recent data (transformed as needed). returns classification.
        transformer (function): function which takes in the most recent data (`buffer`)
            and returns the transformed input the classifer expects.
        action (function): a function which takes in the classification, and
            performs some action.
        calibrator (function): a function which is run on startup to perform
            calibration using `stream`; returns `calibration_info`
            which is used by `classifier` and `transformer`.
        calibration_info: the result of the calibrator, if applicable.
        brain_state: the most recent brain state classification.
        memory_length(int): number of brain states into the past to remember.
        past_states: a list of the past classifications of brain states.
            used in next classification. length is memory_length.
    '''

    def __init__(self, classifier, transformer=None, action=print,
                 calibrator=None, memory_length=10):
        '''
        Initialize a retentive BCI object.

        See class documentation for infromation about the class itself.

        Arguments:
            classifier (function): a function which performs classification on the
                most recent data (transformed as needed). returns class.
            transformer (function): function which takes in the most recent data (`buffer`)
                and returns the transformed input the classifer expects.
            action (function): a function which takes in the classification, and
                performs some action.
            calibrator (function): a function which is run on startup to perform
                calibration using `stream`; returns `calibration_info`
                which is used by `classifier` and `transformer`.
            memory_length (int): number of brain states to remember into past.
        '''

        super().__init__(classifier, transformer, action, calibrator)

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


class automl_BCI(generic_BCI):
    '''
    Implements a Brain-Computer Interface which builds its own classifier
    by training a machine learning model in the calibration stage.

    At calibration, data is recorded for some number of brain-states
    then a machine-learning classifier is trained on the transformed data.

    Internally manages a buffer of the signal given in `stream` and
    continuously performs classification on appropriately transformed
    real-time neural data. At each classification, a corresponding `action`
    is performed.

    Attributes:
        model: a model object which has fit(X, y) and predict(X) methods.
        classifier (function): the model's predictor after training.
            accepts transformed data and returns classification.
        transformer (function): function which takes in the most recent data (`buffer`)
            and returns the transformed input the classifer expects.
        action (function): a function which takes in the classification, and
            performs some action.
        brain_state: the most recent brain state classification.
    '''

    def __init__(self, model, epoch_len, n_states,
                 transformer=None, action=print):
        """
        Initialize an autoML BCI object.

        See class documentation for infromation about the class itself.

        Args:
            model: a model object which has fit(X, y) and predict(X) methods.
            epoch_len (int): the length of the epochs (in # of samples)
                used in training and prediction by the model.
            n_states (int): the number of brain states being classified.
            transformer (function, optional): function which takes in the
                most recent data (`buffer`) and returns the transformed input
                the classifer expects. Defaults to None.
            action (function, optional): a function which takes in the
                classification, and performs some action. Defaults to print.
        """

        super().__init__(None, transformer, action, None)

        self.model = model
        self.epoch_len = epoch_len
        self.n_states = n_states

    def build_model(self, stream, recording_length):
        '''
        records brain signal

        Args:
            stream (neurol.streams object): neurol stream for brain data.
            recording_length (float): length in seconds for the recording of
                each brain state to be used for training the model.
        '''

        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report
        except:
            raise ImportError(
                "could not import scikit-learn. \n"
                "scikit-learn is required for the automl_bci.\n"
                "you can install it using `pip install scikit-learn`")

        from neurol.models import preprocessing

        # Record data for each brain state
        recordings = []

        input('Press Enter to begin...')

        for i in range(self.n_states):
            input(f'Press Enter to begin recording for state {i}')
            # sleep for recording_length while stream accumulates data
            # necessary so no data is used before indicated start of recording
            # get accumulated data
            recording = stream.record_data(recording_length)
            recordings.append(recording)
        print('Done! \n')

        # Epoch data for training model

        # for now, let the inter-window-interval be fixed
        iwi = int(self.epoch_len*0.2)

        epoched_recordings = [preprocessing.epoch(rec, self.epoch_len, iwi)
                              for rec in recordings]

        labels = [np.ones(len(epoched_rec))*i
                  for i, epoched_rec in enumerate(epoched_recordings)]

        X = np.concatenate(epoched_recordings)
        y = np.concatenate(labels)

        # transform data using transformer
        if self.transformer is not None:
            X_transformed = np.array([self.transformer(x) for x in X])
        else:
            X_transformed = X

        # train-test split and shuffle
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=0.20, shuffle=True, stratify=y)

        # fit model to training data
        self.model.fit(X_train, y_train)

        # get classifier from model
        self.classifier = self.model.predict

        # evaluate model
        pred_train = self.classifier(X_train)
        train_report = classification_report(y_train, pred_train)
        print("Performance on training data: ")
        print(train_report)

        pred_test = self.classifier(X_test)
        test_report = classification_report(y_test, pred_test)
        print("Performance on test data: ")
        print(test_report)

    def _update(self, buffer):
        '''transforms, classifies, and acts on data in buffer'''

        latest_epoch = buffer[-self.epoch_len:]  # get latest epoch of data

        # transform buffer for classification
        if self.transformer is not None:
            try:
                clf_input = self.transformer(latest_epoch)
            except TypeError as type_err:
                print(('Got TypeError when calling transformer.\n\n'
                       'Make sure your transformer is a function '
                       'which accepts (latest_epoch) as input.'
                       ))
                raise type_err
        else:
            clf_input = latest_epoch

        # expand dims to prepare for classifier
        clf_input = np.expand_dims(clf_input, axis=0)

        # perform classification
        try:
            self.brain_state = self.classifier(clf_input)[0]
        except TypeError as type_err:
            print(('Got TypeError when calling classifier. \n\n'
                   'Make sure your classifier is a function which '
                   'accepts (clf_input) as inputs.'))
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
            stream (neurol.streams object): neurol stream for brain data.
        '''
        if self.classifier is None:
            print(('Classifier is None.\n'
                   'Make sure you run `build_model` before continueing.'))
            return

        # TODO: implement ending condition?
        running = True  # currently constantly running.

        while running:
            # if new data available, run _update on it
            if stream.update_buffer():
                self._update(stream.buffer)
