'''
Module containing functions for quickly connecting to BCI-related streaming
devices.
'''

import ble2lsl
from ble2lsl.devices import muse2016


def connect_muse():
    '''
    connects to any available muse headset.
    returns ble2lsl.ble2lsl.Streamer object.
    '''
    return ble2lsl.Streamer(muse2016)
