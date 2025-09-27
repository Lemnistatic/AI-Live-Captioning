#!/usr/bin/env python3
import torch, whisper
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
print("Whisper available models:", whisper.available_models() if hasattr(whisper, 'available_models') else 'N/A')
