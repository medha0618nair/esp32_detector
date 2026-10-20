"""
main.py - Entry point: live object detection with spoken announcements.

How the pieces fit together:

    config.py    settings (video source, thresholds, cooldown, ...)
        |
    main.py      opens the stream and runs the loop:
        |          read frame -> detector.process() -> announcer.announce()
        |          -> cv2.imshow() -> check for 'q'
        |
        +-- detector.py  ObjectDetector: YOLO via cvlib + draw_bbox
        +-- tts.py       Announcer: macOS `say` with a per-label cooldown

Run with:  python main.py
"""

import time

import cv2

import config
from detector import ObjectDetector
from tts import Announcer


WEBCAM_PERMISSION_HELP = """
Could not open the Mac camera. macOS needs to allow camera access for the
app you're running this from (VS Code, Terminal, iTerm...):

  1. Open System Settings > Privacy & Security > Camera
  2. Turn ON the switch for that app (e.g. "Visual Studio Code")
  3. Quit that app completely (Cmd+Q) and reopen it, then run main.py again

If the app isn't in the list, reset its permission so macOS asks again:
  VS Code:  tccutil reset Camera com.microsoft.VSCode
  Terminal: tccutil reset Camera com.apple.Terminal
then run main.py again and click "Allow" when asked.

Also close other apps that might be using the camera (FaceTime, Zoom, Photo Booth).
"""

ESP32_HELP = """
Could not open the ESP32-CAM stream {source!r}.
  - Check the board is powered and the Serial Monitor printed "Ready!"
  - The laptop must be on the same Wi-Fi as the ESP32-CAM
  - Close any browser tab showing the stream (only one viewer at a time)
  - If esp32cam.local doesn't work, use the IP from the Serial Monitor
"""


def open_stream(source, permission_wait=30):
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        return cap
    if isinstance(source, str):
        raise SystemExit(ESP32_HELP.format(source=source))

    # On the first run macOS shows an "allow camera access?" popup, but
    # OpenCV gives up straight away instead of waiting for the answer.
    # Keep retrying for a while so clicking "Allow" just works.
    print(f'Waiting up to {permission_wait}s for camera access - click "Allow" if macOS asks...')
    deadline = time.monotonic() + permission_wait
    while time.monotonic() < deadline:
        time.sleep(1)
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            return cap
    raise SystemExit(WEBCAM_PERMISSION_HELP)


def main():
    is_network_stream = isinstance(config.VIDEO_SOURCE, str)
    if is_network_stream and "<ESP32_IP>" in config.VIDEO_SOURCE:
        raise SystemExit("Set ESP32_STREAM_URL in config.py to your ESP32-CAM's real IP first.")

    detector = ObjectDetector(
        backend=config.DETECTOR_BACKEND,
        model=config.YOLO_MODEL if config.DETECTOR_BACKEND == "ultralytics" else config.CVLIB_MODEL,
        confidence=config.CONFIDENCE_THRESHOLD,
        detect_every_n_frames=config.DETECT_EVERY_N_FRAMES,
        names=config.SPOKEN_NAMES,
    )
    announcer = Announcer(
        cooldown=config.SPEECH_COOLDOWN,
        forget_after=config.FORGET_AFTER,
        min_frames=config.MIN_FRAMES_TO_ANNOUNCE,
        voice=config.SPEECH_VOICE,
        rate=config.SPEECH_RATE,
    )

    print(f"Opening video source: {config.VIDEO_SOURCE!r}")
    cap = open_stream(config.VIDEO_SOURCE)
    prev_time = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if not is_network_stream:
                    print("Webcam stopped returning frames; exiting.")
                    break
                # ESP32 streams drop now and then, so reconnect rather than quit.
                print(f"Stream read failed; reconnecting in {config.RECONNECT_DELAY}s...")
                cap.release()
                time.sleep(config.RECONNECT_DELAY)
                cap = cv2.VideoCapture(config.VIDEO_SOURCE)
                continue

            annotated, labels = detector.process(frame)

            if labels is not None:  # None = detection skipped this frame
                spoken = announcer.announce(labels)
                if spoken:
                    print("Announced:", ", ".join(spoken))

            if config.SHOW_FPS:
                now = time.monotonic()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(config.WINDOW_NAME, annotated)

            # waitKey also drives the window's event loop; mask to the low byte for portability.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        announcer.stop()
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # lets macOS actually close the window


if __name__ == "__main__":
    main()
