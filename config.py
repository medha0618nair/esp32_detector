"""
config.py - Global settings for the object detection + TTS system.

Every other module reads its settings from here, so this is the only file
you should need to edit to tune behaviour or switch video sources.
"""

# ---------------------------------------------------------------------------
# Video source
# ---------------------------------------------------------------------------
# True  = use the ESP32-CAM running esp32_cam_stream/esp32_cam_stream.ino
# False = use the Mac's built-in webcam (handy for testing without the board)
USE_ESP32 = False

# The sketch announces itself as "esp32cam.local". If that doesn't connect,
# use the IP address the sketch prints on the Serial Monitor instead, e.g.
# "http://192.168.1.50:81/stream".
ESP32_STREAM_URL = "http://esp32cam.local:81/stream"

VIDEO_SOURCE = ESP32_STREAM_URL if USE_ESP32 else 0

# Network streams sometimes drop frames or disconnect. When a read fails,
# main.py waits this many seconds and then reopens the stream.
RECONNECT_DELAY = 2.0

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
# Minimum confidence (0.0 - 1.0) for a detection to be kept.
CONFIDENCE_THRESHOLD = 0.5

# Which detector to use:
#   "ultralytics" - modern YOLO11, much more accurate, uses the Mac's GPU
#   "cvlib"       - cvlib's older YOLOv3/v4 models (less accurate)
DETECTOR_BACKEND = "ultralytics"

# Model for the "ultralytics" backend. Bigger = more accurate but slower.
# Measured on this Mac's camera (GPU):
#   "yolo26l"  ~12 fps  newest (2026) and most accurate that stays smooth
#   "yolo26m"  ~18 fps  a bit less accurate, smoother
#   "yolo11m"  ~14 fps  previous generation
#   "yolo11s"  ~30 fps, "yolo11n" ~58 fps  fast but less accurate
#   "yolo26x" is the most accurate of all but slower again.
# Weights download into the models/ folder on first run.
YOLO_MODEL = "yolo26l"

# Model for the "cvlib" backend: "yolov4-tiny", "yolov4", "yolov3", or
# "yolov3-tiny". Weights download to ~/.cvlib on first run.
CVLIB_MODEL = "yolov4-tiny"

# Rename YOLO's labels to the words you want spoken and drawn on the boxes.
# Anything not listed keeps its YOLO name. The model knows the 80 COCO
# classes: person, cup, wine glass, bottle, cell phone, book, laptop, ...
SPOKEN_NAMES = {
    "person": "human",       # any face/person in front of the camera
    "cup": "glass",          # YOLO calls a plain drinking glass a "cup" (mugs too)
    "wine glass": "glass",
    "cell phone": "phone",
    "tv": "screen",          # YOLO11 names
    "dining table": "table",
    "tvmonitor": "screen",   # same classes as named by cvlib's models
    "diningtable": "table",
}

# Run detection on every Nth frame and reuse the last result in between.
# Raise this if the video feels laggy (e.g. with the full "yolov4" model).
DETECT_EVERY_N_FRAMES = 1

# ---------------------------------------------------------------------------
# Text-to-speech
# ---------------------------------------------------------------------------
# An object is announced as soon as it comes into view. While it stays in
# view, it is repeated at most once every SPEECH_COOLDOWN seconds.
SPEECH_COOLDOWN = 10.0

# How long (seconds) an object must be out of view before showing it again
# counts as a new appearance and is announced right away.
FORGET_AFTER = 1.5

# An object must be detected in this many frames in a row before it's
# spoken. This filters out one-frame mistakes (e.g. a cup briefly guessed
# as a phone). Higher = fewer wrong announcements but a slightly later one.
MIN_FRAMES_TO_ANNOUNCE = 4

# Optional macOS voice (run `say -v '?'` to list them) and speaking rate
# in words per minute. None = use the system defaults.
SPEECH_VOICE = None
SPEECH_RATE = None

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
WINDOW_NAME = "Object Detector (press 'q' to quit)"
SHOW_FPS = True
