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
CLASS_CATEGORY = {
    PERSON_CLASS_ID: "person",
    CELL_PHONE_CLASS_ID: "phone",
    BOOK_CLASS_ID: "notes",
    LAPTOP_CLASS_ID: "device",
    TV_CLASS_ID: "device",
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
    any secondary device they could be reading from."""

    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.45):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.target_classes = sorted(CLASS_CATEGORY.keys())

    def detect(self, frame):
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            classes=self.target_classes,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            label = CLASS_CATEGORY.get(cls_id, "device")
            detections.append(Detection(cls_id, label, conf, (x1, y1, x2, y2)))
        return detections


def count_label(detections, label):
    return sum(1 for d in detections if d.label == label)


def has_label(detections, label):
    return any(d.label == label for d in detections)
