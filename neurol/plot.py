'''
Module for plotting stream of neural data.
Includes time domain, fourrier transform, and spectrogram live plots.
'''

import numpy as np
from scipy import signal

try:
    import pyqtgraph as pg
except ImportError:
    raise ImportError(
        "pyqtgraph is not installed. \n"
        "pyqtgraph is required for neurol's plotting functionality.\n"
        "you can install it using `pip install pyqtgraph`")


#TODO:
# add filtering

def plot(stream, channels=None, w_size=(1920, 1080)):
    """
    plots data stream. one row per channel.

    Args:
        stream (neurol.streams object): neurol stream for a data source.
        channels: channels to plot. list/tuple of channel indices,
            or dict with indices as keys and names as values.
            Defaults to None (plots all channels w/o names).
        w_size (tuple, optional): initial size of window in pixels.
            Defaults to (1920, 1080).
    """


    # get info about stream
    n_channels = stream.n_channels
    s_rate = stream.sampling_rate
    buffer_length = stream.buffer_length


    ts = np.linspace(-int(buffer_length/s_rate), 0, num=buffer_length)

    # initialize pyqt graph app, window
    win = pg.GraphicsLayoutWidget(show=True)
    win.setWindowTitle('Live Plot')
    app = pg.QtGui.QApplication
    win.resize(w_size[0], w_size[1])

    # initialize plots/curves
    plots = []
    plot_curves = []

    if channels is None:
        channels = list(range(n_channels))

    if isinstance(channels, (list, tuple)):
        for ch_ind in channels:
            plt = win.addPlot(title=f'channel {ch_ind}',
                              labels={'left': 'mV'}
                              )
            plt.setMouseEnabled(x=False, y=False)
            plt.showGrid(x=True)

            plt_curve = plt.plot(y=stream.buffer[:, ch_ind], x=ts)

            plots.append(plt)
            plot_curves.append(plt_curve)

            win.nextRow()

        ch_inds = list(channels)

    elif isinstance(channels, dict):
        for ch_ind, ch_name in channels.items():
            plt = win.addPlot(title=f'{ch_name} channel',
                              labels={'left': 'mV'}
                              )
            plt.setMouseEnabled(x=False, y=False)
            plt.showGrid(x=True)

            plt_curve = plt.plot(y=stream.buffer[:, ch_ind], x=ts)

            plots.append(plt)
            plot_curves.append(plt_curve)

            win.nextRow()

        ch_inds = list(channels.keys())

    else:
        raise ValueError('`channels` argument should be list, tuple, or dict')


    # label bottom plot's x-axis
    plots[-1].setLabel('bottom', 'time', units='s')


    # process initialization events
    app.processEvents()


    # currently always running, TODO: implement ending condition on app close
    running = True

    while running:
        if stream.update_buffer():  # update buffer if new data available
            for ch_ind, plot_curve in zip(ch_inds, plot_curves):
                plot_curve.setData(y=stream.buffer[:, ch_ind], x=ts)

            app.processEvents()

        if not win.isVisible():
            running = False
            app.quit()

def plot_fft(stream, channels=None, w_size=(1920, 1080)):
    """
    plots fourrier transform of data stream from inlet. one row per channel.

    Args:
        stream (neurol.streams object): neurol stream for a data source.
        channels: channels to plot. list/tuple of channel indices,
            or dict with indices as keys and names as values.
            Defaults to None (plots all channels w/o names).
        w_size (tuple, optional): initial size of window in pixels.
            Defaults to (1920, 1080).
    """


    # get info about stream
    n_channels = stream.n_channels
    s_rate = stream.sampling_rate
    buffer_length = stream.buffer_length


    fs = np.fft.fftshift(np.fft.fftfreq(buffer_length, 1/s_rate))

    # initialize pyqt graph app, window
    win = pg.GraphicsLayoutWidget(show=True)
    win.setWindowTitle('Fourrier Transform Live Plot')
    app = pg.QtGui.QApplication
    win.resize(w_size[0], w_size[1])

    # initialize plots/curves
    plots = []
    plot_curves = []

    if channels is None:
        channels = list(range(n_channels))

    if isinstance(channels, (list, tuple)):
        for ch_ind in channels:
            plt = win.addPlot(title=f'channel {ch_ind}')
            plt.setLabel('left', 'voltage-secs', units='Vs') # finalize above

            plt.setMouseEnabled(x=False, y=False)
            plt.showGrid(x=True)

            plt_curve = plt.plot(x=fs)

            plots.append(plt)
            plot_curves.append(plt_curve)

            win.nextRow()

        ch_inds = list(channels)

    elif isinstance(channels, dict):
        for ch_ind, ch_name in channels.items():
            plt = win.addPlot(title=f'{ch_name} channel')
            plt.setLabel('left', 'voltage-secs', units='Vs') # finalize above
            plt.setMouseEnabled(x=False, y=False)
            plt.showGrid(x=True)

            plt_curve = plt.plot(x=fs)

            plots.append(plt)
            plot_curves.append(plt_curve)

            win.nextRow()

        ch_inds = list(channels.keys())

    else:
        raise ValueError('`channels` argument should be list, tuple, or dict')


    # label bottom plot's x-axis
    plots[-1].setLabel('bottom', 'frequency', units='Hz')


    # process initialization events
    app.processEvents()


    running = True
    while running:
        if stream.update_buffer():  # update buffer if new data available
            buff_V = stream.buffer / 1000 # get buffer and convert from mV to V

            # compute fft
            fft = np.fft.fftshift(np.fft.fft(buff_V, axis=0))
            fft_mag = np.abs(fft)

            for ch_ind, plot_curve in zip(ch_inds, plot_curves):
                plot_curve.setData(y=fft_mag[:, ch_ind], x=fs)

            app.processEvents()

        if not win.isVisible():
            running = False
            app.quit()


# NOTE: this can be optimized, currentlythere are redundant computations
# NOTE: works, but spectrogram always seems to produce extreme outliers
# that mess up the graph...

def plot_spectrogram(stream, channels=None, w_size=(1920, 1080)):
    """
    plots spectrogram of data stream from inlet. one row per channel.

    Args:
        stream (neurol.streams object): neurol stream for a data source.
        channels: channels to plot. list/tuple of channel indices,
            or dict with indices as keys and names as values.
            Defaults to None (plots all channels w/o names).
        w_size (tuple, optional): initial size of window in pixels.
            Defaults to (1920, 1080).
    """


    # get info about stream
    n_channels = stream.n_channels
    s_rate = stream.sampling_rate
    buffer_length = stream.buffer_length

    # get format of frequencies and times
    freqs, ts, _ = signal.spectrogram(
        stream.buffer[:, 0], s_rate, mode='magnitude')
    ts = np.max(ts) - ts  # reverse ts s.t. it takes the range [-max_ts, 0]

    # initialize pyqt graph app, window
    win = pg.GraphicsLayoutWidget(show=True)
    win.setWindowTitle('Spectrogram Live Plot')
    app = pg.QtGui.QApplication
    win.resize(w_size[0], w_size[1])

    # interpret image data as row-major to match scipy output
    pg.setConfigOptions(imageAxisOrder='row-major')


    # initialize plots/curves
    plots = []
    plot_imgs = []
    #plot_hists = []

    if channels is None:
        channels = list(range(n_channels))

    # TODO: implement x-axis time range using ts defined above
    # TODO: implement y-axis frequency range using freqs defined above
    if isinstance(channels, (list, tuple)):
        for ch_ind in channels:
            plt = win.addPlot(title=f'channel {ch_ind}')
            plt.setMouseEnabled(x=False, y=False)

            img = pg.ImageItem()
            plt.addItem(img)

            # interactive histogram
            hist = pg.HistogramLUTItem()
            # link the histogram to the image
            hist.setImageItem(img)
            # set histogram gradient
            hist.gradient.restoreState(
                {'mode': 'rgb',
                 'ticks': [(0.5, (0, 182, 188, 255)),
                           (1.0, (246, 111, 0, 255)),
                           (0.0, (75, 0, 113, 255))]})

            win.addItem(hist)


            plots.append(plt)
            plot_imgs.append(img)

            win.nextRow()

        ch_inds = list(channels)


    elif isinstance(channels, dict):
        for ch_ind, ch_name in channels.items():

            plt = win.addPlot(title=f'{ch_name} channel')
            plt.setMouseEnabled(x=False, y=False)

            img = pg.ImageItem()
            plt.addItem(img)

            # interactive histogram
            hist = pg.HistogramLUTItem()
            # link the histogram to the image
            hist.setImageItem(img)
            # set histogram gradient
            hist.gradient.restoreState(
                {'mode': 'rgb',
                 'ticks': [(0.5, (0, 182, 188, 255)),
                           (1.0, (246, 111, 0, 255)),
                           (0.0, (75, 0, 113, 255))]})

            win.addItem(hist)


            plots.append(plt)
            plot_imgs.append(img)

            win.nextRow()

        ch_inds = list(channels.keys())

    else:
        raise ValueError('`channels` argument should be list, tuple, or dict')


    # label bottom plot's x-axis
    plots[-1].setLabel('bottom', 'time', units='s')


    # process initialization events
    app.processEvents()


    running = True
    while running:
        if stream.update_buffer():  # update buffer if new data available

            for ch_ind, plot_img in zip(ch_inds, plot_imgs):
                freqs, _, Sxx = signal.spectrogram(
                    stream.buffer[:, ch_ind], s_rate, mode='magnitude')

                # FIXME: temp fix of outliers
                Sxx_clipped = np.clip(Sxx, 0, 100)
                plot_img.setImage(Sxx_clipped)


            app.processEvents()

        if not win.isVisible():
            running = False
            app.quit()
