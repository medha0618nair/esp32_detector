"""
detector.py - Object detection with OpenCV + YOLO, drawn with cvlib.

main.py creates one ObjectDetector and calls process(frame) on every frame.
process() returns the frame with bounding boxes/labels drawn on it (via
cvlib's draw_bbox) plus the list of detected labels, which main.py hands
to the TTS announcer.

Two detection backends (pick one with DETECTOR_BACKEND in config.py):
  "ultralytics"  modern YOLO11 models; much more accurate, and run on the
                 Mac's GPU (Apple "mps"). The default.
  "cvlib"        cvlib's YOLOv3/v4 models on OpenCV's DNN (CPU). Smaller
                 and simpler, but older and less accurate.
Both know the same 80 everyday object types (COCO dataset).
"""

import importlib.util
import os
import sys
import types

import numpy as np
import requests


def _import_cvlib_object_detection():
    """Import cvlib's object-detection functions.

    `import cvlib` also loads cvlib's gender-detection module, which needs
    TensorFlow (and face detection needs setuptools' pkg_resources).
    Object detection needs neither. If the normal import fails, we register
    an empty `cvlib` package and import only the object_detection submodule.
    That skips cvlib/__init__.py, so TensorFlow never has to be installed.
    """
    try:
        from cvlib.object_detection import detect_common_objects, draw_bbox
        return detect_common_objects, draw_bbox
    except ImportError:
        spec = importlib.util.find_spec("cvlib")
        if spec is None or not spec.submodule_search_locations:
            raise ImportError("cvlib is not installed. Run: pip install -r requirements.txt")

        package = types.ModuleType("cvlib")
        package.__path__ = list(spec.submodule_search_locations)
        sys.modules["cvlib"] = package

        from cvlib.object_detection import detect_common_objects, draw_bbox
        return detect_common_objects, draw_bbox


detect_common_objects, draw_bbox = _import_cvlib_object_detection()

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# cvlib backend: model files
# ---------------------------------------------------------------------------
# cvlib caches model files here and downloads anything missing on first use.
# Its own downloader crashes on gzip-compressed responses and leaves a
# truncated file behind that it then trusts forever, so we fetch the files
# ourselves (same URLs and file names cvlib uses) before cvlib looks for them.
_CVLIB_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".cvlib", "object_detection", "yolo", "yolov3")
_MODEL_FILES = {
    "yolov3": [
        ("yolov3.cfg", "https://github.com/arunponnusamy/object-detection-opencv/raw/master/yolov3.cfg"),
        ("yolov3.weights", "https://pjreddie.com/media/files/yolov3.weights"),
    ],
    "yolov3-tiny": [
        ("yolov3-tiny.cfg", "https://github.com/pjreddie/darknet/raw/master/cfg/yolov3-tiny.cfg"),
        ("yolov3-tiny.weights", "https://pjreddie.com/media/files/yolov3-tiny.weights"),
    ],
    "yolov4": [
        ("yolov4.cfg", "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"),
        ("yolov4.weights", "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"),
    ],
    "yolov4-tiny": [
        ("yolov4-tiny.cfg", "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"),
        ("yolov4-tiny.weights", "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"),
    ],
}
_CLASS_LABELS = ("yolov3_classes.txt", "https://github.com/arunponnusamy/object-detection-opencv/raw/master/yolov3.txt")


def _download(url, path):
    print(f"Downloading {os.path.basename(path)} ...")
    tmp_path = path + ".part"
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    # Rename only once complete, so an interrupted download is retried next run.
    os.replace(tmp_path, path)


def ensure_cvlib_files(model=None):
    """Download cvlib's class labels (needed by draw_bbox) and, if `model`
    is given, that model's YOLO cfg/weights."""
    if model is not None and model not in _MODEL_FILES:
        raise ValueError(f"Unknown cvlib model {model!r}; choose from {sorted(_MODEL_FILES)}")

    os.makedirs(_CVLIB_MODEL_DIR, exist_ok=True)
    files = [_CLASS_LABELS] + (_MODEL_FILES[model] if model else [])
    for file_name, url in files:
        path = os.path.join(_CVLIB_MODEL_DIR, file_name)
        if not os.path.exists(path):
            _download(url, path)


def _make_drawable(labels, colour_source):
    """Let draw_bbox accept labels it doesn't know, such as "human".

    draw_bbox picks each box's colour by looking the label up in cvlib's
    class list, so an unknown name would crash it. Unknown labels are
    added to that list: renamed ones reuse the colour of the YOLO class
    they replace (colour_source maps new name -> original), others get a
    random colour. Called after every detection because cvlib's own
    detector reloads its class list on first use.
    """
    od = sys.modules["cvlib.object_detection"]
    if od.classes is None:
        od.classes = od.populate_class_labels()
    for label in labels:
        if label in od.classes:
            continue
        source = colour_source.get(label)
        if source in od.classes:
            colour = od.COLORS[od.classes.index(source)]
        else:
            colour = np.random.uniform(0, 255, 3)
        # Trim first so colours stay aligned with classes if cvlib reset them.
        od.COLORS = np.vstack([od.COLORS[:len(od.classes)], colour])
        od.classes.append(label)


class ObjectDetector:
    def __init__(self, backend="ultralytics", model="yolo11m", confidence=0.5,
                 detect_every_n_frames=1, names=None):
        self.backend = backend
        self.model = model
        self.confidence = confidence
        self.detect_every_n_frames = max(1, int(detect_every_n_frames))
        self.names = names or {}  # YOLO label -> name to show/speak
        self._colour_source = {new: original for original, new in self.names.items()}

        if backend == "ultralytics":
            ensure_cvlib_files()
            self._load_ultralytics(model)
        elif backend == "cvlib":
            ensure_cvlib_files(model)
        else:
            raise ValueError(f"Unknown DETECTOR_BACKEND {backend!r}; use 'ultralytics' or 'cvlib'")

        self._frame_count = 0
        # Most recent detection result, reused on frames we skip.
        self._last_result = ([], [], [])  # (bboxes, labels, confidences)

    def _load_ultralytics(self, model):
        import torch
        from ultralytics import YOLO

        # Weights (e.g. models/yolo11m.pt) download automatically on first run.
        os.makedirs(os.path.join(PROJECT_DIR, "models"), exist_ok=True)
        self._yolo = YOLO(os.path.join(PROJECT_DIR, "models", model + ".pt"))
        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using {model} on {'the GPU (mps)' if self._device == 'mps' else 'the CPU'}")

    def detect(self, frame):
        """Run YOLO on a frame. Returns (bboxes, labels, confidences)."""
        if self.backend == "ultralytics":
            result = self._yolo.predict(frame, conf=self.confidence,
                                        device=self._device, verbose=False)[0]
            boxes = result.boxes
            bboxes = boxes.xyxy.cpu().numpy().astype(int).tolist()
            labels = [result.names[int(c)] for c in boxes.cls.cpu().numpy()]
            confidences = boxes.conf.cpu().numpy().tolist()
        else:
            bboxes, labels, confidences = detect_common_objects(
                frame, confidence=self.confidence, model=self.model
            )

        labels = [self.names.get(label, label) for label in labels]
        _make_drawable(labels, self._colour_source)
        return bboxes, labels, confidences

    def process(self, frame):
        """Detect objects (every Nth frame) and draw them onto the frame.

        Returns (annotated_frame, labels). `labels` is None on frames where
        detection was skipped, so the announcer only counts fresh results.
        """
        fresh = self._frame_count % self.detect_every_n_frames == 0
        self._frame_count += 1

        if fresh:
            self._last_result = self.detect(frame)

        bboxes, labels, confidences = self._last_result
        annotated = draw_bbox(frame, bboxes, labels, confidences, write_conf=True)
        return annotated, (labels if fresh else None)
