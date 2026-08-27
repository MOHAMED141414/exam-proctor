import time
from collections import defaultdict

import torch
from ultralytics import YOLO

# Standard COCO-80 class ids used by pretrained YOLOv8 weights. Everything
# here is already a COCO class, so no custom training is needed.
PERSON_CLASS_ID = 0
CELL_PHONE_CLASS_ID = 67
BOOK_CLASS_ID = 73
LAPTOP_CLASS_ID = 63
TV_CLASS_ID = 62
MOUSE_CLASS_ID = 64
REMOTE_CLASS_ID = 65
KEYBOARD_CLASS_ID = 66

# COCO id -> the coarse category this system reasons about. Several distinct
# COCO classes collapse into "device" because the alert is the same either
# way: a second screen or input device within reach of the student.
# 'tv' is deliberately absent. It was 17 of the 18 device alerts in a logged
# session, every one of them a monitor on the classroom wall - furniture the
# student cannot read an answer off. Keeping it made the whole device check
# noise. A second screen the student actually uses still registers as 'laptop'.
CLASS_CATEGORY = {
    PERSON_CLASS_ID: "person",
    CELL_PHONE_CLASS_ID: "phone",
    BOOK_CLASS_ID: "notes",
    LAPTOP_CLASS_ID: "device",
    MOUSE_CLASS_ID: "device",
    REMOTE_CLASS_ID: "device",
    KEYBOARD_CLASS_ID: "device",
}

# Colour per category for the overlay boxes (BGR)
CATEGORY_COLOR = {
    "person": (0, 200, 0),
    "phone": (0, 0, 255),
    "notes": (0, 140, 255),
    "device": (255, 0, 200),
}


class Detection:
    def __init__(self, cls_id, label, conf, box):
        self.cls_id = cls_id
        self.label = label          # coarse category, not the raw COCO name
        self.conf = conf
        self.box = box              # (x1, y1, x2, y2)

    @property
    def color(self):
        return CATEGORY_COLOR.get(self.label, (200, 200, 200))


class ObjectDetector:
    """Thin wrapper around a pretrained YOLOv8 model, filtered down to the
    classes this system cares about: the student, phones, printed notes, and
    any secondary device they could be reading from.

    Confidence is gated per category rather than globally. The categories are
    not equally hard: COCO models read printed paper poorly and wall monitors
    easily, so a single threshold either drowns in furniture or never sees a
    note. The model runs at a low floor and each category sets its own bar.
    """

    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.45,
                 category_conf_min=None, imgsz=640):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.category_conf_min = category_conf_min or {}
        self.imgsz = imgsz
        # Pick the GPU explicitly. Left to itself the model stays on the CPU
        # even when CUDA is present and usable, which silently costs ~6x here
        # (34ms vs 222ms per sweep at 640 on this machine).
        self.device = 0 if torch.cuda.is_available() else "cpu"
        # Raising torch's CPU thread count was measured and does nothing here:
        # 4 threads 158/160ms against 8 threads 161/167ms per sweep, fresh
        # process each time. An earlier apparent 201->162ms win turned out to
        # be warm-up in the first timed run, not parallelism. Left at torch's
        # default deliberately - the sweep is memory-bound, not core-starved.
        self.target_classes = sorted(CLASS_CATEGORY.keys())

    def detect(self, frame):
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            device=self.device,
            conf=self.conf_threshold,
            classes=self.target_classes,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = CLASS_CATEGORY.get(cls_id, "device")
            if conf < self.category_conf_min.get(label, self.conf_threshold):
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            detections.append(Detection(cls_id, label, conf, (x1, y1, x2, y2)))
        return detections


def iou(box_a, box_b):
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class StaticBackgroundFilter:
    """Learns which detections are room furniture and drops them.

    A monitor on the wall behind the student is a 'device' by every measure
    YOLO has, and it is there in every single frame - which is exactly what
    distinguishes it from a device the student produced. Anything holding the
    same position throughout the opening window is treated as part of the room.

    'person' is never learned. A second person sitting still is the thing the
    second-person check exists to catch, so suppressing them would turn a
    detection into a blind spot.
    """

    EXEMPT = frozenset({"person"})

    def __init__(self, calibration_sec, iou_threshold, min_seen_ratio):
        self.calibration_sec = calibration_sec
        self.iou_threshold = iou_threshold
        self.min_seen_ratio = min_seen_ratio
        self.started = time.time()
        self.samples = 0                 # calibration sweeps observed
        self.seen = defaultdict(list)    # label -> [[box, hit_count], ...]
        self.static = defaultdict(list)  # label -> [box, ...] (built once)
        self._locked = False

    def calibrating(self):
        return time.time() - self.started < self.calibration_sec

    def remaining(self):
        return max(0.0, self.calibration_sec - (time.time() - self.started))

    def observe(self, detections):
        """Record one calibration sweep, clustering boxes by position."""
        self.samples += 1
        for d in detections:
            if d.label in self.EXEMPT:
                continue
            for entry in self.seen[d.label]:
                if iou(entry[0], d.box) >= self.iou_threshold:
                    entry[1] += 1
                    break
            else:
                self.seen[d.label].append([d.box, 1])

    def _lock(self):
        """Freeze the learned background once calibration ends."""
        for label, entries in self.seen.items():
            for box, hits in entries:
                if self.samples and hits / self.samples >= self.min_seen_ratio:
                    self.static[label].append(box)
        self._locked = True

    def filter(self, detections):
        if not self._locked:
            self._lock()
        kept = []
        for d in detections:
            if d.label not in self.EXEMPT and any(
                iou(box, d.box) >= self.iou_threshold for box in self.static[d.label]
            ):
                continue
            kept.append(d)
        return kept

    def summary(self):
        # The background is built lazily on first filter(), so a caller asking
        # before then would otherwise be told nothing was learned.
        if not self._locked:
            self._lock()
        total = sum(len(v) for v in self.static.values())
        if not total:
            return "no static background learned"
        parts = [f"{label} x{len(boxes)}" for label, boxes in self.static.items() if boxes]
        return f"ignoring {total} static object(s): " + ", ".join(parts)


def count_label(detections, label):
    return sum(1 for d in detections if d.label == label)


def has_label(detections, label):
    return any(d.label == label for d in detections)
