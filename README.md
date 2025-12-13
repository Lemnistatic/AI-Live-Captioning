# StreamCC : AI Live Captioning

This project provides real-time face detection and live captioning using your webcam. It combines OpenCV for face detection, Whisper for speech-to-text transcription, and sounddevice for audio input, delivering synchronized captions with a configurable delay for improved accuracy.

## Features

-   Real-time face detection using OpenCV's Haar cascade classifier.
-   Live audio transcription using OpenAI's Whisper model.
-   Displays captions synchronized with the video feed.
-   Configurable delay to improve caption accuracy.
-   Adjustable audio gain for optimal transcription.

## Requirements

-   Python 3.6+
-   pip

## Dependencies

The project relies on the following Python packages:

-   **opencv-python**: For face detection.
-   **numpy**: For numerical operations.
-   **sounddevice**: For audio input.
-   **torch**: For the Whisper model.
-   **openai-whisper**: For speech-to-text transcription.

Install the required dependencies using pip:

```bash
pip install opencv-python numpy sounddevice torch openai-whisper
