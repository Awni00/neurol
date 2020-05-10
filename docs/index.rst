.. neurol documentation master file, created by
   sphinx-quickstart on Sat May  9 14:30:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:code:`neurol` Documentation
==================================

.. image:: logo/neurol_360dpi.png
    :width: 300px
    :align: center
    :alt: neurol logo

Welcome! :code:`neurol` is a python package for implementing Brain-Computer Interfaces in a modular manner. With the help of tools in the package, you will be able define the behavior of your intended BCI and easily implement it. A :code:`neurol` BCI is defined by a number of components:

- A :code:`classifier` which decodes brain data into some kind of 'brain-state'
- An :code:`action` which provides feedback depending on the decoded 'brain-state'
- An optional :code:`calibrator` which runs at startup and modifies the operation of the BCI
- An optional :code:`transformer` which transforms the current :code:`buffer` of data into the form expected by the :code:`classifier`

The :code:`neurol` BCI manages an incoming stream of brain data and uses the above user-defined functions to run a brain-computer interface.

The package includes generic utility functions to aid in creating the :code:`classifier`'s, :code:`transfromer`'s, and :code:`calibrator`'s for common BCI use-cases. It also comes prepackaged with a growing list of trained machine learning models for common BCI classification tasks.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   neurol


Indices
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
