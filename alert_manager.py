import csv
import os
import time
from collections import defaultdict, deque

import cv2
import numpy as np

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

    def __init__(self, config, seat="", shared_log=False):
        self.cfg = config
        self.seat = seat
        # Every seat writes to one CSV, so only the first instance creates the
        # header and the seat column is what separates the rows afterwards.
        self.shared_log = shared_log
        self.mar_history = deque()              # (timestamp, mar), this seat only
        self.look_away_events = deque()
        self.gaze_events = deque()
        self.talk_events = deque()

        self.phone_streak = 0
        self.notes_streak = 0
        self.device_streak = 0
        self.second_person_streak = 0

        self.last_person_seen = time.time()
        self.absent_sweeps = 0                  # consecutive sweeps finding nobody
        self.ever_occupied = False              # has anyone ever sat here?
        self.currently_away = False
        self.away_since = None
        self.prev_gray = None                   # previous frame, for the frozen-feed check
        self.people_now = 0                     # live count, drawn on the panel
        self.people_max = 0                     # high-water mark for the summary
        self.mar_std = 0.0                      # last computed, for the overlay

        # start time of each currently-held sustained condition
        self.holding = {}

        self.active_alerts = {}                 # label -> banner expiry time
        self.last_fired = {}                    # label -> last fire time
        self.counts = defaultdict(int)          # label -> total this session
        self.session_start = time.time()
        self.current_frame = None               # set each loop, used for snapshots

        self.HEADER = ["timestamp", "seat", "type", "detail", "snapshot"]
        if self.cfg.ENABLE_CSV_LOG:
            self._ensure_log()

        if getattr(self.cfg, "SAVE_SNAPSHOTS", False):
            os.makedirs(self.cfg.SNAPSHOT_DIR, exist_ok=True)

    # --- plumbing -------------------------------------------------------

    def _ensure_log(self):
        """Create the log, or roll an older one aside if its columns differ.

        The seat column was added after earlier sessions had already written
        four-column rows. Appending to that file produces a CSV whose columns
        mean different things depending on which run wrote the row, which no
        reader can interpret - so the old file is kept, under a new name.
        """
        path = self.cfg.LOG_PATH
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(self.HEADER)
            return
        with open(path, newline="") as f:
            existing = next(csv.reader(f), [])
        if existing == self.HEADER:
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base, ext = os.path.splitext(path)
        os.rename(path, f"{base}_pre-seats_{stamp}{ext}")
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(self.HEADER)

    def set_frame(self, frame):
        """Hand the manager the current frame so an alert can save evidence."""
        self.current_frame = frame

    def _snapshot(self, label, ts):
        if not getattr(self.cfg, "SAVE_SNAPSHOTS", False) or self.current_frame is None:
            return ""
        safe = "".join(c if c.isalnum() else "_" for c in label).strip("_")
        seat = f"{self.seat}_" if self.seat else ""
        name = f"{ts.replace(':', '-').replace(' ', '_')}_{seat}{safe}.jpg"
        path = os.path.join(self.cfg.SNAPSHOT_DIR, name)
        cv2.imwrite(path, self.current_frame)
        return path

    def _log(self, label, detail, snapshot):
        if self.cfg.ENABLE_CSV_LOG:
            with open(self.cfg.LOG_PATH, "a", newline="") as f:
                csv.writer(f).writerow(
                    [time.strftime("%Y-%m-%d %H:%M:%S"), self.seat,
                     label, detail, snapshot]
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

        # Looking down is its own check, so look-away must not also claim it.
        # While it did, the lower look-away threshold always won the race and
        # the look-down rule was unreachable: every downward tilt was reported
        # as a glance away instead. Turning aside and looking up still count.
        looking_down = (pitch * self.cfg.LOOK_DOWN_PITCH_SIGN) > self.cfg.LOOK_DOWN_PITCH_DEG
        is_away = abs(yaw) > self.cfg.YAW_THRESHOLD_DEG or (
            abs(pitch) > self.cfg.PITCH_THRESHOLD_DEG and not looking_down
        )

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
        if self._sustained("look_down", looking_down, self.cfg.LOOK_DOWN_MIN_DURATION_SEC):
            self._fire(LOOK_DOWN, f"pitch {pitch:.0f} deg held")

    def update_gaze(self, gaze_offset, head_forward, face_width=0.0, yaw=0.0):
        """Attention off screen, measured whichever way the image allows.

        Close enough to resolve the eyes, this is the original iris check: it
        catches the student who keeps their head still and moves only their
        eyes, which head pose alone cannot see.

        Too far for that, the irises are noise and the check would either never
        fire or fire constantly, so it falls back to head direction - a
        sustained moderate turn, below the angle look-away wants. Weaker, but
        it measures something real instead of pretending to measure gaze.
        """
        now = time.time()
        use_iris = face_width >= self.cfg.GAZE_MIN_FACE_PX

        if use_iris:
            # Iris geometry is foreshortened once the head turns far, and the
            # look-away check already owns that case, so stand down - but fall
            # through rather than returning. Returning here left banked events
            # unpruned, and an alert that had already met its threshold sat
            # waiting for the head to come back before it could fire.
            if not head_forward:
                off = False
            else:
                off = gaze_offset > self.cfg.GAZE_OFFSET_THRESHOLD
        else:
            # The head-direction fallback must NOT stand down at large yaw -
            # a big turn is the clearest possible evidence of looking away.
            # Suppressing it there left the check live only in the narrow
            # 15-28 degree band, which is why it still never fired.
            off = abs(yaw) > self.cfg.GAZE_HEAD_YAW_DEG
        if self._sustained("gaze", off, self.cfg.GAZE_MIN_DURATION_SEC):
            self.gaze_events.append(now)
            self.holding["gaze"] = now      # restart so one long stare is one event

        self._prune(self.gaze_events, self.cfg.GAZE_WINDOW_SEC, now)
        if len(self.gaze_events) >= self.cfg.GAZE_EVENT_COUNT:
            self._fire(GAZE_OFF, f"{len(self.gaze_events)}x in {self.cfg.GAZE_WINDOW_SEC}s")
            self.gaze_events.clear()

    def update_talking(self, mar):
        """Takes the raw mouth-aspect ratio and keeps its own rolling window.

        The variance of mouth opening is what distinguishes talking from a
        mouth that merely happens to be open, and that window has to belong to
        one student - a history shared across seats would average three
        people's mouths into one meaningless signal.
        """
        now = time.time()
        self.mar_history.append((now, mar))
        while self.mar_history and now - self.mar_history[0][0] > self.cfg.MAR_STD_WINDOW_SEC:
            self.mar_history.popleft()
        values = [m for _, m in self.mar_history]
        self.mar_std = float(np.std(values)) if len(values) > 2 else 0.0

        if self.mar_std > self.cfg.MAR_STD_THRESHOLD:
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
        """How many people occupy THIS seat. One is expected; two is the
        violation - somebody has leaned into this student's space.

        Bodies and faces are counted separately and the larger wins: YOLO
        merges two bodies into one box at close range, where the face count
        still sees both, and a person turned away has a body but no face.
        """
        people = max(person_count, face_count)
        self.people_now = people
        self.people_max = max(self.people_max, people)

        self.second_person_streak = self.second_person_streak + 1 if people > 1 else 0
        if self.second_person_streak >= self.cfg.SECOND_PERSON_CONSEC_FRAMES:
            self._fire(SECOND_PERSON,
                       f"{people} people in this seat "
                       f"(bodies={person_count} faces={face_count})")
            self.second_person_streak = 0

    def update_hand_near_face(self, near):
        if self._sustained("hand_face", near, self.cfg.HAND_NEAR_FACE_MIN_SEC):
            self._fire(HAND_NEAR_FACE)
            self.holding.pop("hand_face", None)

    def update_presence(self, person_seen):
        """Empty chair, working camera. Callers must only reach this with a feed
        that passed every tamper check, so a covered lens can never be reported
        as the student walking out - those are different events.

        Absence has to clear two independent bars: no sighting for ABSENCE_SEC,
        and several detection sweeps in a row finding nobody. The timer alone
        trusted a run of unlucky frames; requiring consecutive empty sweeps
        means a detector dropout has to persist to count as an empty room.
        """
        now = time.time()
        if person_seen:
            self.last_person_seen = now
            self.absent_sweeps = 0
            self.ever_occupied = True
            return

        # A seat nobody ever sat in is an empty chair, not a student who walked
        # out. Without this an unused seat re-fires absence at the cooldown
        # floor for the whole session - 35 alerts from one empty seat in a
        # ten-minute run, drowning every real finding.
        if not self.ever_occupied:
            return

        self.absent_sweeps += 1
        long_enough = now - self.last_person_seen > self.cfg.ABSENCE_SEC
        confirmed = self.absent_sweeps >= self.cfg.ABSENCE_CONSEC_SWEEPS
        if long_enough and confirmed:
            self._fire(LEFT_FRAME, f"absent {int(now - self.last_person_seen)}s, "
                                   f"{self.absent_sweeps} empty sweeps")
            self.last_person_seen = now  # do not re-fire every single frame
            self.absent_sweeps = 0

    def update_tamper(self, frame):
        """Four ways the feed can stop showing the room, each with its own
        signature: covered by something opaque (dark), pointed at a blank
        surface (flat), covered by a hand pressed to the lens (bright and
        textured, but completely out of focus), or dead (identical frames).

        The blur test is what catches a hand: dark and flat both miss it
        entirely, because a hand lit from the side is neither.
        """
        if frame is None:
            blocked = True
            reason = "no frame"
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Half-size for the Laplacian: this runs on every frame, unlike the
            # throttled YOLO and hand checks, and blur survives downsampling.
            small = cv2.resize(gray, None, fx=0.5, fy=0.5)

            # Brightness and uniformity are measured on `small` as well, not
            # on the full colour frame. numpy's std() over 1920x1080x3 upcasts
            # to float64 and costs 48ms by itself; the whole method measured
            # 61.5ms against 6ms for this version, on EVERY frame, which made
            # the cheapest check in the system by far the most expensive.
            # Reusing one probe also means all four tests judge the same
            # pixels. TAMPER_BLUR_LAPVAR is calibrated at this half resolution
            # - change the scale and that threshold silently becomes wrong.
            mean, std = cv2.meanStdDev(small)

            if mean[0][0] < self.cfg.TAMPER_DARK_MEAN:
                reason = "feed dark"
            elif std[0][0] < self.cfg.TAMPER_FLAT_STD:
                reason = "feed flat"
            elif cv2.Laplacian(small, cv2.CV_32F).var() < self.cfg.TAMPER_BLUR_LAPVAR:
                reason = "lens covered or defocused"
            elif self._frame_frozen(small):
                reason = "feed frozen"
            else:
                reason = ""
            blocked = bool(reason)

        if blocked:
            # Absence is measured from the last confirmed sighting, and nothing
            # can be sighted through a blocked lens. Without this the timer keeps
            # running while covered and fires a bogus "left frame" on uncover.
            self.last_person_seen = time.time()
            self.absent_sweeps = 0

        if self._sustained("tamper", blocked, self.cfg.TAMPER_MIN_SEC):
            self._fire(CAMERA_BLOCKED, reason)
        return blocked

    def _frame_frozen(self, gray_small):
        """True when this frame is essentially identical to the previous one.
        A stalled capture keeps handing back the last good frame rather than
        failing, so 'nothing changed at all' is the only tell."""
        prev, self.prev_gray = self.prev_gray, gray_small
        if prev is None or prev.shape != gray_small.shape:
            return False
        return float(cv2.absdiff(prev, gray_small).mean()) < self.cfg.TAMPER_FROZEN_DIFF

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
        head = f"seat {self.seat}" if self.seat else "session"
        lines = [f"{head}: {total} incidents in {mins:.1f} min "
                 f"(most people at once: {self.people_max})"]
        for label in ALERT_ORDER:
            if self.counts[label]:
                lines.append(f"    {label:<26} {self.counts[label]}")
        if not total:
            lines.append("    (nothing recorded)")
        return "\n".join(lines)
