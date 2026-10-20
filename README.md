# ESP32-CAM Object Announcer

Hold an object in front of the camera and your Mac says what it is out loud
("human", "glass", "phone", "book", ...).

```
ESP32-CAM (Arduino sketch) --Wi-Fi video--> Mac (main.py: YOLO + "say") --> speaker
```

The camera can be the Mac's built-in webcam (for testing) or an ESP32-CAM
streaming over Wi-Fi. Object detection runs on the Mac with YOLO (YOLO26 via
Ultralytics, using the Mac's GPU). It recognises the 80 everyday object types
of the COCO dataset. Speech uses macOS's built-in `say` command.

## Project files

| File | What it does |
|---|---|
| `config.py` | All settings: camera source, model, confidence, speech timing, renamed labels |
| `main.py` | Entry point: reads frames, runs detection, triggers speech, shows the window |
| `detector.py` | YOLO detection and drawing boxes with cvlib's `draw_bbox` |
| `tts.py` | Speech via `say`, with a cooldown and a filter for one-frame mistakes |
| `esp32_cam_stream/esp32_cam_stream.ino` | Arduino sketch that turns an AI-Thinker ESP32-CAM into a Wi-Fi camera |

## Setup (macOS)

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

The YOLO weights download into `models/` on the first run.

## Run with the Mac webcam

In `config.py` set `USE_ESP32 = False`, then:

```bash
python main.py
```

Allow camera access if macOS asks. Press `q` in the video window to quit.

## Run with the ESP32-CAM

1. Open `esp32_cam_stream/esp32_cam_stream.ino` in the Arduino IDE.
2. Fill in `WIFI_SSID` and `WIFI_PASSWORD` (2.4 GHz network), and **don't
   commit them**.
3. Board: **AI Thinker ESP32-CAM** (esp32 by Espressif package). Upload.
4. Open the Serial Monitor at 115200 baud and wait for `Ready!`.
5. In `config.py` set `USE_ESP32 = True`. If `esp32cam.local` doesn't
   connect, set `ESP32_STREAM_URL` to the IP printed on the Serial Monitor,
   e.g. `http://192.168.1.50:81/stream`.
6. Run `python main.py`. The Mac must be on the same Wi-Fi network.

## Tuning (`config.py`)

- `YOLO_MODEL`: `yolo26l` (default, most accurate that stays smooth), `yolo26m`
  (faster), `yolo26x` (most accurate, slower).
- `SPOKEN_NAMES`: rename labels, e.g. `"person": "human"`, `"cup": "glass"`.
- `MIN_FRAMES_TO_ANNOUNCE`: frames in a row an object must be seen before
  it's spoken. Raise it to reduce wrong announcements.
- `SPEECH_COOLDOWN`: seconds before an object that stays in view is repeated.
