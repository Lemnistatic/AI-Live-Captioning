import cv2
import numpy as np
import threading
import time
import sounddevice as sd
import torch
import queue
import sys
from collections import deque
from datetime import datetime, timedelta

# Whisper
import whisper

# Globals
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
FACE_HOLD_SECONDS = 5.0
SAMPLERATE = 16000
BLOCKSIZE = 8000
GAIN = 4.0  # Increased from 1.5 to 4.0
DEVICE = None  # Let's auto-select the default input device instead of hardcoding 1

face_box = None
last_face_seen = 0
audio_queue = queue.Queue()

# Delay system
DELAY_SECONDS = 5
frame_buffer = deque(maxlen=DELAY_SECONDS * 30)  # Assuming 30fps
caption_timeline = deque()  # Store (timestamp, caption) pairs

print("Loading Whisper model...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
whisper_model = whisper.load_model("base")
print("Whisper model loaded successfully!")

# Check audio devices
print("\nAvailable audio devices:")
print(sd.query_devices())

def get_best_audio_device():
    devices = sd.query_devices()
    print("\nAvailable Audio Devices:")
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # Input device
            print(f"[{i}] {device['name']}")
    
    # Try to find a good default
    default_device = sd.query_devices(kind='input')
    device_id = default_device['index']
    print(f"\nSelected device {device_id}: {default_device['name']}")
    return device_id

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio callback status: {status}", file=sys.stderr)
    
    # Debug audio levels
    current_level = np.abs(indata).mean()
    if current_level > 0.001:  # Only print when there's significant audio
        print(f"Audio level: {current_level:.4f}")
    
    # Apply higher gain
    amplified_audio = indata.copy() * GAIN
    current_time = time.time()
    audio_queue.put((current_time, amplified_audio))

def audio_thread():
    global DEVICE
    print("Starting audio thread...")
    try:
        # Auto-select best input device
        DEVICE = get_best_audio_device()
        
        with sd.InputStream(
            samplerate=SAMPLERATE,
            blocksize=BLOCKSIZE,
            device=DEVICE,
            channels=1,
            dtype="float32",
            callback=audio_callback
        ):
            print("Audio stream started successfully!")
            print(f"Using input device {DEVICE}")
            print("Speak to test - you should see audio levels in console")
            while True:
                time.sleep(0.1)
    except Exception as e:
        print(f"Audio thread error: {e}")

def transcription_thread():
    print("Transcription thread started!")
    
    # Rolling audio buffer for transcription
    audio_buffer = deque(maxlen=int(SAMPLERATE * 6))  # Keep 6 seconds of audio
    last_transcription = 0
    transcription_interval = 1.0  # Transcribe every 1 second
    
    while True:
        try:
            # Collect audio data with timestamps
            current_time = time.time()
            
            # Add new audio to buffer
            while not audio_queue.empty():
                timestamp, audio_data = audio_queue.get()
                for sample in audio_data.flatten():
                    audio_buffer.append((timestamp, sample))
            
            # Only transcribe at intervals
            if current_time - last_transcription < transcription_interval:
                time.sleep(0.1)
                continue
            
            # Get recent audio (last 3 seconds)
            if len(audio_buffer) < SAMPLERATE * 2:  # Need at least 2 seconds
                time.sleep(0.1)
                continue
            
            # Extract recent audio samples
            recent_samples = []
            transcription_timestamp = None
            
            # Get audio from the last 3 seconds
            cutoff_time = current_time - 3.0
            for timestamp, sample in list(audio_buffer)[-int(SAMPLERATE * 3):]:
                if timestamp >= cutoff_time:
                    if transcription_timestamp is None:
                        transcription_timestamp = timestamp
                    recent_samples.append(sample)
            
            if len(recent_samples) < SAMPLERATE:  # Need at least 1 second
                time.sleep(0.1)
                continue
                
            audio_data = np.array(recent_samples, dtype=np.float32)
            audio_level = np.abs(audio_data).mean()
            
            # Skip if too quiet - much stricter threshold
            if audio_level < 0.02:  # Increased from 0.01 to 0.02
                print(f"Audio too quiet (level: {audio_level:.4f}), skipping transcription")
                last_transcription = current_time
                continue
            
            print(f"Transcribing audio from {transcription_timestamp:.1f} (level: {audio_level:.4f})")
            
            # Transcribe with strict settings to prevent hallucination
            result = whisper_model.transcribe(
                audio_data,
                fp16=False,
                language="en",
                task="transcribe",
                verbose=False,
                no_speech_threshold=0.8,    # Much higher threshold (was 0.6)
                logprob_threshold=-0.5,     # Much stricter confidence (was -1.0)
                compression_ratio_threshold=2.4,  # Detect repetitive text
                temperature=0.0,            # No randomness
                condition_on_previous_text=False,  # Don't use previous context
            )
            
            last_transcription = current_time
            
            # Process results
            if result["segments"]:
                full_text = " ".join([segment["text"].strip() for segment in result["segments"]]).strip()
                
                if full_text and len(full_text) > 2:
                    print(f"✓ Caption at {transcription_timestamp:.1f}: '{full_text}'")
                    
                    # Store caption with its timestamp for delayed playback
                    caption_timeline.append((transcription_timestamp, full_text))
                    
                    # Keep only recent captions (last 30 seconds)
                    while caption_timeline and caption_timeline[0][0] < current_time - 30:
                        caption_timeline.popleft()
                        
        except Exception as e:
            print(f"Transcription error: {e}")
            import traceback
            traceback.print_exc()

def get_caption_for_time(target_time):
    """Get the most recent caption for a specific timestamp"""
    if not caption_timeline:
        return ""
    
    # Find the most recent caption before or at target_time
    best_caption = ""
    for timestamp, caption in caption_timeline:
        if timestamp <= target_time:
            best_caption = caption
        else:
            break  # Captions are stored chronologically
            
    return best_caption

def video_thread():
    global face_box, last_face_seen
    
    print("Starting video capture...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    print("Camera opened successfully!")
    
    # Fill initial buffer
    print("Buffering frames for delay...")
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        current_time = time.time()
        
        # Add current frame to buffer with timestamp
        frame_buffer.append((current_time, frame.copy()))
        
        # Process delayed frame
        if len(frame_buffer) > 0:
            # Get the oldest frame (most delayed)
            delayed_time, delayed_frame = frame_buffer[0]
            
            # Only start showing delayed video after buffer is full
            if current_time - start_time < DELAY_SECONDS:
                # Still buffering - show current frame with buffer status
                display_frame = frame.copy()
                remaining = DELAY_SECONDS - (current_time - start_time)
                cv2.putText(display_frame, f"Buffering... {remaining:.1f}s remaining", 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_frame, "Start speaking now for testing!", 
                           (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow("AI Live Captioning", display_frame)
            else:
                # Show delayed frame with synchronized captions
                
                # Face detection on delayed frame
                gray = cv2.cvtColor(delayed_frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

                if len(faces) > 0:
                    best = max(faces, key=lambda f: f[2] * f[3])
                    face_box = (int(best[0]), int(best[1]), int(best[2]), int(best[3]))
                    last_face_seen = current_time
                elif face_box is not None:
                    if current_time - last_face_seen > FACE_HOLD_SECONDS:
                        face_box = None

                # Draw face box
                if face_box:
                    (x, y, w, h) = face_box
                    cv2.rectangle(delayed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Get caption that was spoken at the time of this delayed frame
                delayed_caption = get_caption_for_time(delayed_time)
                
                # Draw caption
                if delayed_caption:
                    # Create background rectangle for better text visibility
                    caption_text = delayed_caption[:100]  # Limit length
                    lines = []
                    
                    # Split long captions into multiple lines
                    words = caption_text.split()
                    current_line = ""
                    for word in words:
                        if len(current_line + " " + word) < 50:
                            current_line += (" " + word if current_line else word)
                        else:
                            if current_line:
                                lines.append(current_line)
                            current_line = word
                    if current_line:
                        lines.append(current_line)
                    
                    # Draw background and text for each line
                    for i, line in enumerate(lines):
                        y_pos = FRAME_HEIGHT - 60 + (i * 25)
                        text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        
                        # Background rectangle
                        cv2.rectangle(delayed_frame, 
                                     (15, y_pos - 20), 
                                     (25 + text_size[0], y_pos + 5), 
                                     (0, 0, 0), -1)
                        
                        # Text
                        cv2.putText(delayed_frame, line, (20, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                # Add status info
                delay_info = f"Video Delay: {DELAY_SECONDS}s | Frame Time: {delayed_time:.1f}"
                cv2.putText(delayed_frame, delay_info, (20, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                
                # Show caption count
                caption_info = f"Captions in timeline: {len(caption_timeline)}"
                cv2.putText(delayed_frame, caption_info, (20, 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

                cv2.imshow("AI Live Captioning", delayed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()

# Start everything
if __name__ == "__main__":
    print("=== AI Live Captioning - Real Audio + Delayed Video ===")
    print("Instructions:")
    print("1. Audio is captured and processed in REAL-TIME")
    print("2. Video is delayed by 5 seconds")  
    print("3. Captions appear on delayed video showing what was said at that time")
    print("4. Start speaking immediately - captions will appear on delayed video")
    print("5. Press 'q' to quit")
    print("=" * 60)
    
    try:
        # Start all threads
        threading.Thread(target=audio_thread, daemon=True).start()
        threading.Thread(target=transcription_thread, daemon=True).start()
        video_thread()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Main thread error: {e}")
        import traceback
        traceback.print_exc()