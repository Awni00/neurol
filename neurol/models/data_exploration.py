'''
Module containing functions to study and analyze neural signals,
especially to provide insights for building machine learning models
to perform classification relevant to Brain-Computer Interface applications.
'''

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError(
        "matplotlib is not installed. \n"
        "matplotlib is required for the data_exploration module.\n"
        "you can install it using `pip install matplotlib`")

try:
    from sklearn.decomposition import PCA, FastICA
except ImportError:
    raise ImportError(
        "scikit-learn is not installed. \n"
        "scikit-learn is required for some data_exploration functionality.\n"
        "you can install it using `pip install scikit_learn`")

import numpy as np


def plot_signal(signal, sampling_rate, signal_type=None, ch_names=None,
                event_timestamps=None, **plt_kwargs):
    '''
    Plots signal.

    Arguments:
        signal: signal as array of shape [n_samples, n_channels].
        sr(float): sampling rate in samples per second.
        signal_type: (optional) gives a title for the y-axis.
        ch_names: (optional) array of names for each channel (used for legend).
        event_timestamps: (optional) 1-D array of times at which an
            event/stimulus occured.
        **plt_kwargs: matplotlib keyword args
    '''

    time = np.linspace(0, len(signal)/sampling_rate, len(signal))

    plt.plot(time, signal, **plt_kwargs)

    if ch_names is not None:
        plt.legend(ch_names)

    if event_timestamps is not None:
        for timestamp in event_timestamps:
            plt.axvline(timestamp, ls='--', lw=1, c='violet')

    plt.xlabel("time [s]")
    plt.ylabel(signal_type)


def plot_grid(signals, num_signals=None, sampling_rate=1, cols=4,
              fig_size=(10, 6), sharey=True, sharex=True, random=True,
              fig_axes=None, show=True):
    '''
    Plot an (optionally random) set of signals [epochs] in a grid
    from a larger array of given signals.

    Arguments:
        signals: array of signals to plot from (num_signals, num_samples).
        num_signals(int): the number of siganls to plot.
        sampling_rate(float): sampling rate of signals.
        cols(int): the number of columns in the grid.
        fig_size: tuple (x,y) of figure size in inches.
        sharey(bool): whether to share scale on y-axis (see matplotlib).
        sharex(bool): whether to share scale on x-axis (see matplotlib).
        random(bool): whether to choose signals randomly or just use
            the first num_signals.
        fig_axes: optionally, an existing tuple of (fig,axes) to plot on
            (see matplotlib) rather creating new one.
        show(bool): whether to show plot inline.

    Returns:
        fig, axes: matplotlib figure and axes with sample of signals
            plotted in a grid
    '''

    # if num_signals to be plotted is not given, plot all signals
    if num_signals is None:
        num_signals = signals.shape[0]

    rows = int(np.ceil(num_signals / cols))

    if fig_axes is None:
        fig, axes = plt.subplots(
            rows, cols, sharey=sharey, sharex=sharex, figsize=fig_size)
    else:
        fig, axes = fig_axes

    if random:
        # choose a random set of signals to plot
        sampled_signals = signals[np.random.choice(
            range(len(signals)), num_signals, replace=False)]
    else:
        sampled_signals = signals[:num_signals]  # choose the first num

    for row in range(rows):
        for col in range(cols):
            if (row*cols + col) >= num_signals:
                break
            sampled_signal = sampled_signals[row*cols+col]

            t_max = len(sampled_signal) / sampling_rate
            time = np.linspace(0, t_max, len(sampled_signal))
            axes[row][col].plot(time, sampled_signal)

    if show:
        plt.show()
    return fig, axes


def stim_triggered_average(signal, sampling_rate, timestamps, duration_before,
                           duration_after, plot=False):
    '''
    Inspired by the computational neuroscience concept of the
    spike-triggered average, this function computes the average
    signal characteristic around known events.

    Arguments:
        signal: signal as an array of shape [samples, channels].
        sr(float): sampling rate of the signal.
        timestamps: array of floats containing the timestamps for each event.
        duration_before: the duration to be considered before each event.
        duration_after: the duration to be considered after each event.
        plot(optional): whether or not to plot the stim_triggered_average.

    Returns:
        stim_triggered_average: average signal characteristic around event.
        relative_time: relative time of each sample in stim_triggered_average
            with respect to event.

    '''

    stim_indices = (timestamps*sampling_rate).astype(int)

    ind_before = int(sampling_rate * duration_before)
    ind_after = int(sampling_rate * duration_after)

    sta = np.mean(
        [signal[i-ind_before:i+ind_after] for i in stim_indices], axis=0)
    relative_time = np.linspace(-duration_before,
                                duration_after, len(sta))

    if plot:
        plt.plot(relative_time, sta)
        plt.axvline(0, c='violet', ls='--', lw=2)

    return sta, relative_time


def plot_PCA(epochs, sampling_rate=1, n_components=None, return_PCA=False,
             PCA_kwargs=None, plot_grid_kwargs=None):
    '''
    performs principal component analysis and plots principal components
    of epochs of a signal.

    Arguments:
        epochs: array of epochs (n_epochs, n_samples).
        sr(float): sampling rate.
        num_components(int): number of components to use.
            If none is passed, all are used.
        return_PCA(bool): whether to return the independent components.
        PCA_kwargs(dict): dictionary containing kwargs for PCA function
            (see scikit-learn).
        plot_grid_kwargs(dict): dictionary containing kwargs for
            plot_grid function.

    '''

    if not PCA_kwargs:
        PCA_kwargs = {}
    pca = PCA(n_components=n_components, **PCA_kwargs)
    principle_components = pca.fit_transform(epochs.T).T

    if not plot_grid_kwargs:
        plot_grid_kwargs = {}
    plot_grid(principle_components, sampling_rate=sampling_rate,
              random=False, **plot_grid_kwargs)

    if return_PCA:
        return principle_components


def plot_ICA(epochs, sampling_rate=1, n_components=None, return_ICA=False,
             FastICA_kwargs=None, plot_grid_kwargs=None):
    '''
    performs independent component analysis and plots independent components of
    epochs of a signal.

    Arguments:
        epochs: array of epochs (n_epochs, n_samples).
        sr(float): sampling rate.
        num_components(int): number of components to use.
            If none is passed, all are used.
        return_ICA(bool): whether to return the independent components.
        FastICA_kwargs(dict): dictionary containing kwargs for FastICA
            function (see scikit-learn).
        plot_grid_kwargs(dict): dictionary containing kwargs for
            plot_grid function.

    '''

    if not FastICA_kwargs:
        FastICA_kwargs = {}
    ICA = FastICA(n_components=n_components, **FastICA_kwargs)
    independent_components = ICA.fit_transform(epochs.T).T

    if not plot_grid_kwargs:
        plot_grid_kwargs = {}
    plot_grid(independent_components, sampling_rate=sampling_rate,
              random=False, **plot_grid_kwargs)

    if return_ICA:
        return independent_components
