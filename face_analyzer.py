import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

# Generic 3D face model points (mm) used for solvePnP head-pose estimation
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Left eye, left corner
    (225.0, 170.0, -135.0),    # Right eye, right corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0),   # Right mouth corner
], dtype=np.float64)

# Matching MediaPipe FaceMesh landmark indices
LM = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye": 33,
    "right_eye": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 78, 308

# Iris landmarks only exist when refine_landmarks=True
LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER = 468, 473
LEFT_EYE_OUTER, LEFT_EYE_INNER = 33, 133
RIGHT_EYE_INNER, RIGHT_EYE_OUTER = 362, 263

# Face-side landmarks, used as the "ear" anchors for hand-near-face
FACE_LEFT_EDGE, FACE_RIGHT_EDGE = 234, 454


class _Pt:
    """A landmark remapped from a zone crop into full-frame normalized space."""
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y


def _normalize_angle(a):
    """Fold a Euler angle into [-90, 90) so a given sign always means the same
    direction, whatever branch decomposeProjectionMatrix happened to return.

    Subtracting 180 once is not enough. These angles arrive anywhere on the
    circle, and a value near 358 folds to 178 - still outside the range, and
    still wrong. Observed live: a head looking straight ahead reported
    pitch 178, which made "looking down" (pitch < -14) impossible to satisfy
    and "looking away" (|pitch| > 20) permanently true. Two checks, one
    silently broken and one firing constantly, from the same missing wrap.

    Modular arithmetic handles every branch and is idempotent, so re-applying
    it to an already-folded angle changes nothing.
    """
    return ((a + 90.0) % 180.0) - 90.0


class FaceResult:
    """Metrics for ONE face. In multi-student mode the analyzer returns one of
    these per face and the caller attributes each to a seat."""

    def __init__(self, face_found, face_count=0, yaw=0.0, pitch=0.0, mar=0.0,
                 mar_std=0.0, gaze_offset=0.0, nose_2d=None, face_width=0.0,
                 anchors=None, face_box=None):
        self.face_found = face_found
        self.face_count = face_count
        self.yaw = yaw
        self.pitch = pitch
        self.mar = mar
        self.mar_std = mar_std              # filled in per seat by AlertManager
        self.gaze_offset = gaze_offset      # 0 = looking straight ahead
        self.nose_2d = nose_2d
        self.face_width = face_width        # px, used to scale distances
        self.anchors = anchors or []        # ear/mouth points for hand checks
        self.face_box = face_box
        self.seat = None                    # set when found via a zone crop

    @property
    def center_x(self):
        """Horizontal centre, used to decide which seat this face occupies."""
        if not self.face_box:
            return 0.0
        return (self.face_box[0] + self.face_box[2]) / 2.0


class FaceAnalyzer:
    """Head pose (yaw/pitch) for gaze direction, iris offset for eye-only
    gaze, and mouth-aspect-ratio variance as a lightweight talking detector.

    Tracks up to max_faces so a second person leaning into frame is caught;
    the largest face is treated as the student for all per-student metrics."""

    def __init__(self, max_faces=6, zones=0, crop_bottom=1.0):
        self.crop_bottom = crop_bottom
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,          # required for the iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # One mesh per zone. MediaPipe carries tracking state between calls, so
        # feeding one instance three different crops of the same frame would
        # have it chase a face that appears to teleport.
        self.zone_meshes = [
            mp.solutions.face_mesh.FaceMesh(
                max_num_faces=2,            # the student, plus an intruder
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            for _ in range(zones)
        ]

    @staticmethod
    def _face_span(lm, w, h):
        xs = [p.x * w for p in lm]
        ys = [p.y * h for p in lm]
        return min(xs), min(ys), max(xs), max(ys)

    def _gaze_offset(self, lm, w):
        """How far the irises sit from centred, averaged over both eyes.
        0.0 is dead centre, ~0.5 is pinned to one corner. This is head-pose
        blind, so it catches eyes-only glances that yaw/pitch never see."""
        offsets = []
        for iris, c1, c2 in (
            (LEFT_IRIS_CENTER, LEFT_EYE_OUTER, LEFT_EYE_INNER),
            (RIGHT_IRIS_CENTER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER),
        ):
            x_iris = lm[iris].x * w
            x_lo, x_hi = sorted((lm[c1].x * w, lm[c2].x * w))
            span = x_hi - x_lo
            if span > 1e-6:
                ratio = (x_iris - x_lo) / span
                offsets.append(abs(ratio - 0.5))
        return float(np.mean(offsets)) if offsets else 0.0

    def analyze(self, frame):
        """Every face in the frame, each with its own metrics.

        This used to analyse only the largest face and merely count the rest,
        which was correct when extra faces meant cheating. With one student per
        seat every face is somebody being proctored, so each gets the full
        treatment and the caller decides which seat it belongs to.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return []

        faces = results.multi_face_landmarks
        face_count = len(faces)
        out = []
        for f in faces:
            r = self._measure(f.landmark, w, h, face_count)
            if r is not None:
                out.append(r)
        return out

    def analyze_zones(self, frame, seat_count):
        """Run face detection on each seat's crop instead of the whole frame.

        MediaPipe downscales its input to roughly 192px before looking for a
        face, so a 50px face in a 1280px frame arrives as about 7px and is
        simply not found. Cropping to one seat first means that same face is
        50px inside a ~426px strip, which survives the downscale.

        Landmarks are mapped back to full-frame coordinates BEFORE any pose
        maths runs: solvePnP uses frame width as the focal length, so measuring
        inside a crop would silently change the camera intrinsics and skew
        every yaw and pitch reading.
        """
        h, w = frame.shape[:2]
        y1 = max(1, int(h * self.crop_bottom))
        out = []
        for i, mesh in enumerate(self.zone_meshes[:seat_count]):
            x0 = int(w * i / seat_count)
            x1 = int(w * (i + 1) / seat_count) if i < seat_count - 1 else w
            crop = frame[0:y1, x0:x1]
            if crop.size == 0:
                continue
            res = mesh.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                continue
            cw, ch = x1 - x0, y1
            for f in res.multi_face_landmarks:
                mapped = [_Pt((p.x * cw + x0) / w, (p.y * ch) / h)
                          for p in f.landmark]
                r = self._measure(mapped, w, h, len(res.multi_face_landmarks))
                if r is not None:
                    r.seat = i
                    out.append(r)
        return out

    def _measure(self, lm, w, h, face_count):
        """Head pose, gaze and mouth opening for a single face's landmarks."""
        fx1, fy1, fx2, fy2 = self._face_span(lm, w, h)

        image_points = np.array([
            (lm[LM["nose_tip"]].x * w, lm[LM["nose_tip"]].y * h),
            (lm[LM["chin"]].x * w, lm[LM["chin"]].y * h),
            (lm[LM["left_eye"]].x * w, lm[LM["left_eye"]].y * h),
            (lm[LM["right_eye"]].x * w, lm[LM["right_eye"]].y * h),
            (lm[LM["left_mouth"]].x * w, lm[LM["left_mouth"]].y * h),
            (lm[LM["right_mouth"]].x * w, lm[LM["right_mouth"]].y * h),
        ], dtype=np.float64)

        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        ok, rotation_vec, _ = cv2.solvePnP(
            MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, np.zeros((3, 1))))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        pitch, yaw, _roll = [_normalize_angle(float(a)) for a in euler_angles]

        # Mouth aspect ratio: vertical opening over mouth width
        top = np.array([lm[MOUTH_TOP].x * w, lm[MOUTH_TOP].y * h])
        bottom = np.array([lm[MOUTH_BOTTOM].x * w, lm[MOUTH_BOTTOM].y * h])
        left = np.array([lm[MOUTH_LEFT].x * w, lm[MOUTH_LEFT].y * h])
        right = np.array([lm[MOUTH_RIGHT].x * w, lm[MOUTH_RIGHT].y * h])
        horizontal = np.linalg.norm(left - right)
        mar = float(np.linalg.norm(top - bottom) / horizontal) if horizontal > 0 else 0.0

        # mar_std is deliberately NOT computed here. It is the variance of mouth
        # opening over a rolling window, and each student needs their own
        # window - one shared history would blend three people's mouths into a
        # single signal. The per-seat AlertManager owns it now.
        anchors = [
            (lm[FACE_LEFT_EDGE].x * w, lm[FACE_LEFT_EDGE].y * h),
            (lm[FACE_RIGHT_EDGE].x * w, lm[FACE_RIGHT_EDGE].y * h),
            (lm[MOUTH_TOP].x * w, lm[MOUTH_TOP].y * h),
        ]

        nose_2d = (int(image_points[0][0]), int(image_points[0][1]))
        return FaceResult(
            face_found=True,
            face_count=face_count,
            yaw=yaw,
            pitch=pitch,
            mar=mar,
            gaze_offset=self._gaze_offset(lm, w),
            nose_2d=nose_2d,
            face_width=float(fx2 - fx1),
            anchors=anchors,
            face_box=(int(fx1), int(fy1), int(fx2), int(fy2)),
        )

    def close(self):
        self.face_mesh.close()
        for m in self.zone_meshes:
            m.close()


class HandAnalyzer:
    """Hand landmark tracking, used only to answer one question: is a hand up
    at the ear or mouth? That covers an earpiece being adjusted, a whispered
    exchange, and a hand cupped to hide the lips from the camera."""

    def __init__(self, max_hands=6):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=max_hands,
            model_complexity=0,             # fastest variant, enough for proximity
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def analyze(self, frame):
        """Returns a flat list of (x, y) hand landmark points in pixels."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            return []
        return [
            (p.x * w, p.y * h)
            for hand in results.multi_hand_landmarks
            for p in hand.landmark
        ]

    def close(self):
        self.hands.close()


def hand_near_face(hand_points, anchors, face_width, ratio):
    """True when any hand landmark sits within `ratio` face-widths of an ear
    or the mouth. Scaling by face width keeps the rule distance-invariant, so
    the student leaning back does not change what counts as near."""
    if not hand_points or not anchors or face_width <= 0:
        return False
    limit = ratio * face_width
    for hx, hy in hand_points:
        for ax, ay in anchors:
            if (hx - ax) ** 2 + (hy - ay) ** 2 <= limit ** 2:
                return True
    return False
