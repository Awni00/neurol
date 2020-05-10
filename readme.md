<p align='center' >
    <img src='https://github.com/Awni00/neurol/blob/dev/docs/logo/neurol_360dpi.png?raw=true' alt='neurol logo' width='400'/>
</p>

<p align='center'>
    <a href='https://www.python.org/downloads/release'>
  	    <img alt="Python 3.6+" src='https://img.shields.io/badge/python-3.6+-blue.svg'/>
    </a>
    <a href="https://pypi.org/project/neurol/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/neurol" alt="PyPI version">
    </a>
    <a href='https://neurol.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/neurol/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href=https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project>
        <img alt='PR welcome' src='https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?'/>
    </a>
    <a href='https://github.com/Awni00/neurol/blob/dev/LICENSE'>
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/neurol"/>
    </a>
</p>


Neurol is a python package for implementing Brain-Computer Interfaces in a modular manner. With the help of tools in this package, you will be able define the behavior of your intended BCI and easily implement it. A `neurol` BCI is defined by a number of components:

- A `classifier` which decodes brain data into some kind of 'brain-state'
- An `action` which provides feedback depending on the decoded 'brain-state'
- An optional `calibrator` which runs at startup and modifies the operation of the BCI
- An optional `transformer` which transforms the current `buffer` of data into the form expected by the `classifier`

The `neurol` BCI manages an incoming stream of brain data and uses the above user-defined functions to run a brain-computer interface.

The package includes generic utility functions to aid in creating `classifier`'s, `transfromer`'s, and `calibrator`'s for common BCI use-cases. It also comes prepackaged with a growing list of trained machine learning models for common BCI classification tasks.

# Installation

`neurol` can be easily installed using `pip`:

```
$ pip install neurol
```

# Documentation

Please find `neurol`'s documentation <a href='https://neurol.readthedocs.io/'>here</a>.

<!--this is linking to the examples folder in the `dev` branch. switch to master branch at some point-->
You can also find example notebooks in the <a href='https://github.com/Awni00/neurol/tree/dev/examples'>examples</a> directory.

# Contact

If you have questions or would like to discuss this package, please don't hesitate to contact me.

Awni Altabaa - awni.altabaa@queensu.ca / awnyaltabaa@gmail.com