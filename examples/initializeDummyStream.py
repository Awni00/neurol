import ble2lsl
from ble2lsl.devices import muse2016

print("Dummy stream starting. Run BCI script in new terminal window!")
device = muse2016
dummy_outlet = ble2lsl.Dummy(device)