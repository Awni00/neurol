'''
Module containing functions for quickly connecting to BCI-related streaming
devices.
'''

from pylsl import StreamInlet, resolve_byprop


def connect_muse():
    '''
    connects to any available muse headset.
    returns ble2lsl.ble2lsl.Streamer object.
    '''

    try:
        import ble2lsl
        from ble2lsl.devices import muse2016
    except ImportError:
        raise ImportError(
            "ble2lsl is not installed. \n"
            "ble2lsl is used to connect to muse devices.\n"
            "you can install it using `pip install ble2lsl`")

    return ble2lsl.Streamer(muse2016)


def get_lsl_EEG_inlets():
    '''resolves all EEG lsl streams and returns their inlets in an array.'''

    stream_infos = resolve_byprop("type", "EEG")

    inlets = []

    for stream_info in stream_infos:
        inlets.append(StreamInlet(stream_info, ))

    return inlets
