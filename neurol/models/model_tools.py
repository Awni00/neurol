'''
Module for managing the models which come pre-packaged with neurol.
Includes functionality for importing and using the models.
'''

import os

try:
    from tensorflow.keras.models import load_model
except ImportError:
    raise ImportError(
        "tensorflow is not installed. \n"
        "tensorflow is required for some using neurol's models.\n"
        "you can install it using `pip install tensorflow`")

# Note: currently all models are keras .h5 models.
# Think about organization of this module as new models are added



# get path to module to compute path to models
here = os.path.dirname(os.path.realpath(__file__))

def get_model(model_name):
    '''
    gets the specified trained model.

    Arguments:
        model_name(str): name of model.
            See documentation for list of available models.

    Returns:
        model: trained model.of
    '''

    path_to_model = here + '/' + model_name + '.h5'
    model = load_model(path_to_model)
    return model


def get_predictor(model_name):
    '''
    gets the predictor for the specified model.

    Arguments:
        model_name(str): name of model.
            See documentation for list of available models.

    Returns:
        predictor: predictor of trained model.
    '''

    model = get_model(model_name)
    predictor = model.predict
    return predictor
