from neurol import connect_device
from neurol import streams
from neurol import plot

# connect a device

# get the pylsl inlet (lsl device is used)
inlet = connect_device.get_lsl_EEG_inlets()[0]

# create a neurol stream object
stream = streams.lsl_stream(inlet, buffer_length=4096)


# plot! (uncomment and try them all!)
plot.plot(stream)
# plot.plot_fft(stream)
# plot.plot_spectrogram(stream)
