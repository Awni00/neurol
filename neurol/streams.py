'''
module for handling streams of data from different sources
'''

import time
import numpy as np


# TODO: add functionality to specify units and scaling factors

class lsl_stream:
    """
    A generalized stream object for an lsl data source.

    Manages a buffer of data and makes it available.
    Used by neurol.BCI and neurol.plot.
    """

    def __init__(self, pylsl_inlet, buffer_length=2048):
        """
        initialize an lsl_stream object.

        Args:
            pylsl_inlet (pylsl.pylsl.StreamInlet): inlet of connected lsl device
            buffer_length (int, optional): length of data buffer.
                Defaults to 2048.
        """
        self.inlet = pylsl_inlet

        # get number of channels and sampling rate
        info = pylsl_inlet.info()
        self.n_channels = info.channel_count()
        self.sampling_rate = info.nominal_srate()

        # initialize buffer
        self.buffer_length = buffer_length
        self.buffer = np.zeros((self.buffer_length, self.n_channels))

        # open stream
        self.inlet.open_stream()

    def get_data(self, max_samples=2048):
        """
        gets latest data.
        """

        if self.inlet.samples_available():
            # get latest data
            chunk, _ = self.inlet.pull_chunk(max_samples=max_samples)

            return chunk

    def record_data(self, duration):
        """
        records from stream for some duration of time.

        Args:
            duration (float): length of recording in seconds.
        """

        max_samples = int(self.sampling_rate * duration)

        print(f'Recording for {duration} seconds...')
        # sleep for recording_length while stream accumulates data
        # necessary so no data is used before indicated start of recording
        time.sleep(duration)
        # get accumulated data
        recording = self.get_data(max_samples=max_samples)

        return recording

    def update_buffer(self):
        """
        updates buffer with most recent available data.

        Returns:
            [bool]: True if new data available, False if not.
        """

        if self.inlet.samples_available():
            # get latest data
            chunk, _ = self.inlet.pull_chunk(max_samples=self.buffer_length)

            # append to buffer
            self.buffer = np.append(self.buffer, np.array(chunk), axis=0)

            # clip buffer to buffer_length (keep most recent data)
            self.buffer = self.buffer[-self.buffer_length:]

            return True

        else:
            return False


    def close(self):
        """
        closes the pylsl inlet stream
        """

        self.inlet.close_stream()
