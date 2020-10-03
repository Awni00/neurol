from neurol import connect_device
from neurol import streams
from neurol import plot

# connect a device
# here we use a muse (already connected to computer)
# but use whatever you have

# get the pylsl inlet (specific to muses)
inlet = connect_device.get_EEG_inlets()[0]

# create a neurol stream object
muse_stream = streams.lsl_stream(inlet, buffer_length=4096)


# plot! (uncomment and try them all!)
plot.plot(muse_stream)
# plot.plot_fft(muse_stream)
# plot.plot_spectrogram(muse_stream)
