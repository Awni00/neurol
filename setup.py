import setuptools


with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neurol",
    version="0.0.5",
    author="Awni",
    author_email="awni.altabaa@queensu.ca",
    description="A package for modularly implenting Brain-Computer Interfaces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="BCI Brain-Computer Interface Neurotechnology Neuroscience EEG",
    install_requires=[
            "numpy",
            "pylsl",
            "scipy",
    ],
    include_package_data=True,
    url='https://github.com/Awni00/neurol',
    project_urls={'Documentation': 'https://neurol.readthedocs.io/',
        'Source':'https://github.com/Awni00/neurol',
        'Tracker':'https://github.com/Awni00/neurol/issues'},
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
