#!/usr/bin/env python3
import sounddevice as sd
for idx, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] > 0:
        print(f"[{idx}] {dev['name']} (inputs: {dev['max_input_channels']}, default SR: {int(dev['default_samplerate'])})")
