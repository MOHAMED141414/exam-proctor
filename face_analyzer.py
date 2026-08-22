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


def _normalize_angle(a):
    """decomposeProjectionMatrix returns Euler angles that wrap near +-180.
    Fold them back into a signed range so a negative pitch always means the
    same direction instead of flipping when the head crosses a wrap point."""
    if a > 90:
        a -= 180
    elif a < -90:
        a += 180
    return a


class FaceResult:
    def __init__(self, face_found, face_count=0, yaw=0.0, pitch=0.0, mar=0.0,
                 mar_std=0.0, gaze_offset=0.0, nose_2d=None, face_width=0.0,
                 anchors=None, face_box=None):
        self.face_found = face_found
        self.face_count = face_count
        self.yaw = yaw
        self.pitch = pitch
        self.mar = mar
        self.mar_std = mar_std
        self.gaze_offset = gaze_offset      # 0 = looking straight ahead
        self.nose_2d = nose_2d
        self.face_width = face_width        # px, used to scale distances
        self.anchors = anchors or []        # ear/mouth points for hand checks
        self.face_box = face_box


class FaceAnalyzer:
    """Head pose (yaw/pitch) for gaze direction, iris offset for eye-only
    gaze, and mouth-aspect-ratio variance as a lightweight talking detector.

    Tracks up to max_faces so a second person leaning into frame is caught;
    the largest face is treated as the student for all per-student metrics."""

    def __init__(self, mar_window_sec=1.5, max_faces=3):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,          # required for the iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mar_window_sec = mar_window_sec
        self.mar_history = deque()  # (timestamp, mar)

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
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return FaceResult(face_found=False, face_count=0)

        faces = results.multi_face_landmarks
        face_count = len(faces)

        # The student is whichever face is largest. A second person leaning
        # in from behind sits further from the lens and so measures smaller.
        def area(f):
            x1, y1, x2, y2 = self._face_span(f.landmark, w, h)
            return (x2 - x1) * (y2 - y1)

        lm = max(faces, key=area).landmark
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
            return FaceResult(face_found=False, face_count=face_count)

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

        now = time.time()
        self.mar_history.append((now, mar))
        while self.mar_history and now - self.mar_history[0][0] > self.mar_window_sec:
            self.mar_history.popleft()
        mar_values = [m for _, m in self.mar_history]
        mar_std = float(np.std(mar_values)) if len(mar_values) > 2 else 0.0

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
            mar_std=mar_std,
            gaze_offset=self._gaze_offset(lm, w),
            nose_2d=nose_2d,
            face_width=float(fx2 - fx1),
            anchors=anchors,
            face_box=(int(fx1), int(fy1), int(fx2), int(fy2)),
        )

    def close(self):
        self.face_mesh.close()


class HandAnalyzer:
    """Hand landmark tracking, used only to answer one question: is a hand up
    at the ear or mouth? That covers an earpiece being adjusted, a whispered
    exchange, and a hand cupped to hide the lips from the camera."""

    def __init__(self, max_hands=2):
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
