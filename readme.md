# Neurol

Neurol is a python package for implementing Brain-Computer Interfaces in a modular manner. With the help of tools in the package, the user is able to define the intended behavior of the BCI. This includes:

- A `classifier` which decodes brain data into some kind of 'brain-state'
- An `action` which provides feedback depending on the decoded 'brain-state'
- An optional `calibrator` which runs at startup and modifies the operation of the BCI
- An optional `transformer` which transforms the current `buffer` of data into the form expected by the `classifier`

A generic BCI wrapper manages an incoming stream of brain data and uses the above user-defined functions to run a brain-computer interface.

The package includes generic utility functions to aid in creating `classifier`'s, `transfromer`'s, and `calibrator`'s for typical BCI use-cases. It also comes prepackaged with a growing list of trained machine learning models for common BCI classification tasks.