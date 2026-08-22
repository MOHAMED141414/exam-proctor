import csv
import os
import time
from collections import defaultdict, deque

import cv2

# Every alert this system can raise, in the order the on-screen panel lists
# them. Keeping this as one ordered table means the panel, the CSV and the
# end-of-session summary can never drift out of sync with each other.
LOOK_AWAY = "LOOKING AWAY REPEATEDLY"
LOOK_DOWN = "LOOKING DOWN (NOTES)"
GAZE_OFF = "GAZE OFF SCREEN"
TALKING = "TALKING DETECTED"
PHONE = "PHONE / DEVICE VISIBLE"
NOTES = "NOTES / BOOK VISIBLE"
EXTRA_DEVICE = "EXTRA DEVICE VISIBLE"
SECOND_PERSON = "SECOND PERSON PRESENT"
HAND_NEAR_FACE = "HAND AT EAR / MOUTH"
LEFT_FRAME = "STUDENT LEFT FRAME"
CAMERA_BLOCKED = "CAMERA BLOCKED"

ALERT_ORDER = [
    LOOK_AWAY, LOOK_DOWN, GAZE_OFF, TALKING, PHONE, NOTES,
    EXTRA_DEVICE, SECOND_PERSON, HAND_NEAR_FACE, LEFT_FRAME, CAMERA_BLOCKED,
]


class AlertManager:
    """Turns raw per-frame signals (head angle, iris offset, mouth motion,
    detections, hand positions) into discrete alerts. Each alert has its own
    rule for what separates a real violation from noise:

      - repeat-in-window : N events inside a rolling window (looking away,
                           gaze off screen, talking)
      - sustained        : the condition holds continuously for N seconds
                           (looking down, hand at ear, camera blocked)
      - consecutive      : N detections in a row (phone, notes, device,
                           second person)
      - timeout          : nothing seen for N seconds (left frame)

    Every alert additionally passes through a per-label cooldown, so a
    condition that stays true cannot machine-gun the log.
    """

    def __init__(self, config):
        self.cfg = config
        self.look_away_events = deque()
        self.gaze_events = deque()
        self.talk_events = deque()

        self.phone_streak = 0
        self.notes_streak = 0
        self.device_streak = 0
        self.second_person_streak = 0

        self.last_person_seen = time.time()
        self.currently_away = False
        self.away_since = None

        # start time of each currently-held sustained condition
        self.holding = {}

        self.active_alerts = {}                 # label -> banner expiry time
        self.last_fired = {}                    # label -> last fire time
        self.counts = defaultdict(int)          # label -> total this session
        self.session_start = time.time()
        self.current_frame = None               # set each loop, used for snapshots

        if self.cfg.ENABLE_CSV_LOG and not os.path.exists(self.cfg.LOG_PATH):
            with open(self.cfg.LOG_PATH, "w", newline="") as f:
                csv.writer(f).writerow(["timestamp", "type", "detail", "snapshot"])

        if getattr(self.cfg, "SAVE_SNAPSHOTS", False):
            os.makedirs(self.cfg.SNAPSHOT_DIR, exist_ok=True)

    # --- plumbing -------------------------------------------------------

    def set_frame(self, frame):
        """Hand the manager the current frame so an alert can save evidence."""
        self.current_frame = frame

    def _snapshot(self, label, ts):
        if not getattr(self.cfg, "SAVE_SNAPSHOTS", False) or self.current_frame is None:
            return ""
        safe = "".join(c if c.isalnum() else "_" for c in label).strip("_")
        name = f"{ts.replace(':', '-').replace(' ', '_')}_{safe}.jpg"
        path = os.path.join(self.cfg.SNAPSHOT_DIR, name)
        cv2.imwrite(path, self.current_frame)
        return path

    def _log(self, label, detail, snapshot):
        if self.cfg.ENABLE_CSV_LOG:
            with open(self.cfg.LOG_PATH, "a", newline="") as f:
                csv.writer(f).writerow(
                    [time.strftime("%Y-%m-%d %H:%M:%S"), label, detail, snapshot]
                )

    def _fire(self, label, detail=""):
        now = time.time()
        # Per-label cooldown. Without this, a condition that stays true just
        # re-accumulates its events and fires again a few frames later.
        if now - self.last_fired.get(label, 0.0) < self.cfg.ALERT_COOLDOWN_SEC:
            return
        self.last_fired[label] = now
        self.counts[label] += 1
        self.active_alerts[label] = now + self.cfg.ALERT_DISPLAY_SEC
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self._log(label, detail, self._snapshot(label, ts))

    def _sustained(self, key, condition, min_sec):
        """True once `condition` has held continuously for min_sec."""
        now = time.time()
        if not condition:
            self.holding.pop(key, None)
            return False
        started = self.holding.setdefault(key, now)
        return now - started >= min_sec

    @staticmethod
    def _prune(events, window, now):
        while events and now - events[0] > window:
            events.popleft()

    # --- individual checks ----------------------------------------------

    def update_head_pose(self, yaw, pitch):
        now = time.time()
        is_away = abs(yaw) > self.cfg.YAW_THRESHOLD_DEG or abs(pitch) > self.cfg.PITCH_THRESHOLD_DEG

        if is_away:
            if not self.currently_away:
                self.currently_away = True
                self.away_since = now
            elif now - self.away_since >= self.cfg.LOOK_AWAY_MIN_DURATION_SEC:
                self.look_away_events.append(now)
                self.away_since = now  # reset so a long turn does not spam events
        else:
            self.currently_away = False
            self.away_since = None

        self._prune(self.look_away_events, self.cfg.LOOK_AWAY_WINDOW_SEC, now)
        if len(self.look_away_events) >= self.cfg.LOOK_AWAY_EVENT_COUNT:
            self._fire(LOOK_AWAY,
                       f"{len(self.look_away_events)}x in {self.cfg.LOOK_AWAY_WINDOW_SEC}s")
            self.look_away_events.clear()

        # Looking down is treated separately from looking away: a student
        # reading notes on the desk holds a steady downward tilt rather than
        # producing the repeated turn-and-return the look-away rule counts.
        looking_down = (pitch * self.cfg.LOOK_DOWN_PITCH_SIGN) > self.cfg.LOOK_DOWN_PITCH_DEG
        if self._sustained("look_down", looking_down, self.cfg.LOOK_DOWN_MIN_DURATION_SEC):
            self._fire(LOOK_DOWN, f"pitch {pitch:.0f} deg held")

    def update_gaze(self, gaze_offset, head_forward):
        """Eyes-only glance detection. Only meaningful while the head is
        roughly forward: at a steep yaw the iris geometry is foreshortened
        and the head-pose check already owns that case."""
        now = time.time()
        if not head_forward:
            self.holding.pop("gaze", None)
            return

        off = gaze_offset > self.cfg.GAZE_OFFSET_THRESHOLD
        if self._sustained("gaze", off, self.cfg.GAZE_MIN_DURATION_SEC):
            self.gaze_events.append(now)
            self.holding["gaze"] = now      # restart so one long stare is one event

        self._prune(self.gaze_events, self.cfg.GAZE_WINDOW_SEC, now)
        if len(self.gaze_events) >= self.cfg.GAZE_EVENT_COUNT:
            self._fire(GAZE_OFF, f"{len(self.gaze_events)}x in {self.cfg.GAZE_WINDOW_SEC}s")
            self.gaze_events.clear()

    def update_talking(self, mar_std):
        now = time.time()
        if mar_std > self.cfg.MAR_STD_THRESHOLD:
            self.talk_events.append(now)

        self._prune(self.talk_events, self.cfg.TALK_WINDOW_SEC, now)
        if len(self.talk_events) >= self.cfg.TALK_EVENT_COUNT:
            self._fire(TALKING, f"{len(self.talk_events)}x in {self.cfg.TALK_WINDOW_SEC}s")
            self.talk_events.clear()

    def update_objects(self, phone_seen, notes_seen, device_seen):
        self.phone_streak = self.phone_streak + 1 if phone_seen else 0
        if self.phone_streak >= self.cfg.PHONE_CONSEC_FRAMES:
            self._fire(PHONE)
            self.phone_streak = 0

        self.notes_streak = self.notes_streak + 1 if notes_seen else 0
        if self.notes_streak >= self.cfg.NOTES_CONSEC_FRAMES:
            self._fire(NOTES)
            self.notes_streak = 0

        self.device_streak = self.device_streak + 1 if device_seen else 0
        if self.device_streak >= self.cfg.DEVICE_CONSEC_FRAMES:
            self._fire(EXTRA_DEVICE)
            self.device_streak = 0

    def update_person_count(self, person_count, face_count):
        """A second body OR a second face both mean someone else is there.
        Faces catch a helper leaning in over the shoulder at close range,
        where YOLO often merges the two bodies into one box."""
        extra = max(person_count, face_count) > 1
        self.second_person_streak = self.second_person_streak + 1 if extra else 0
        if self.second_person_streak >= self.cfg.SECOND_PERSON_CONSEC_FRAMES:
            self._fire(SECOND_PERSON, f"persons={person_count} faces={face_count}")
            self.second_person_streak = 0

    def update_hand_near_face(self, near):
        if self._sustained("hand_face", near, self.cfg.HAND_NEAR_FACE_MIN_SEC):
            self._fire(HAND_NEAR_FACE)
            self.holding.pop("hand_face", None)

    def update_presence(self, person_seen):
        now = time.time()
        if person_seen:
            self.last_person_seen = now
        elif now - self.last_person_seen > self.cfg.ABSENCE_SEC:
            self._fire(LEFT_FRAME, f"absent {int(now - self.last_person_seen)}s")
            self.last_person_seen = now  # do not re-fire every single frame

    def update_tamper(self, frame):
        """Covered lens, capped phone, or a dead DroidCam feed all present the
        same way: a frame with almost no brightness or almost no variation."""
        if frame is None:
            blocked = True
        else:
            blocked = float(frame.mean()) < self.cfg.TAMPER_DARK_MEAN or \
                      float(frame.std()) < self.cfg.TAMPER_FLAT_STD
        if self._sustained("tamper", blocked, self.cfg.TAMPER_MIN_SEC):
            self._fire(CAMERA_BLOCKED, "feed dark or frozen")
        return blocked

    # --- output ---------------------------------------------------------

    def _expire(self):
        now = time.time()
        self.active_alerts = {k: v for k, v in self.active_alerts.items() if v > now}

    def get_active_banner(self):
        self._expire()
        return " | ".join(self.active_alerts.keys()) if self.active_alerts else None

    def get_status(self):
        """(label, is_active, total_count) for every check, panel order."""
        self._expire()
        return [(label, label in self.active_alerts, self.counts[label])
                for label in ALERT_ORDER]

    def summary(self):
        total = sum(self.counts.values())
        mins = (time.time() - self.session_start) / 60.0
        lines = [f"Session length : {mins:.1f} min",
                 f"Total incidents: {total}"]
        for label in ALERT_ORDER:
            if self.counts[label]:
                lines.append(f"  {label:<26} {self.counts[label]}")
        if not total:
            lines.append("  (no incidents recorded)")
        return "\n".join(lines)
