"""
Central configuration for the exam proctoring system.

Single camera: the iPhone running DroidCam. The laptop webcam is not used.
Edit these values to match your setup before running main.py.
"""

# --- Camera source (DroidCam only) ---
# The laptop webcam is deliberately not used.
#
# Windows enumerates cameras differently per capture backend, so the backend
# is pinned rather than left to OpenCV. Verified on this machine:
#     MSMF  index 0 = laptop "HD camera"
#     MSMF  index 1 = DroidCam            <-- the one we want
#     DSHOW index 1 = laptop, index 2 = OBS Virtual Camera
# Leaving the backend unset makes index 1 ambiguous between phone and laptop.
CAMERA_SOURCE = 1
CAMERA_BACKEND = "MSMF"
CAMERA_NAME = "iPhone / DroidCam"

# Alternative: skip the DroidCam Client and pull the stream over WiFi using
# the IP:port the DroidCam phone app shows. Set CAMERA_SOURCE to the URL,
# e.g. "http://192.168.1.50:4747/video" - the backend setting is then ignored.

# Position the phone so it sees the student face AND the desk surface.
# Every check runs on this one feed, so framing matters more than it did with
# two cameras: a face-only crop gives up notes- and phone-on-desk detection.

# --- Display ---
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600
PANEL_WIDTH = 320            # right-hand live checklist
WINDOW_NAME = "Exam Proctor - press 'q' to quit"

# --- Performance ---
DETECT_EVERY_N_FRAMES = 3     # YOLO cadence
HANDS_EVERY_N_FRAMES = 3      # MediaPipe Hands cadence (costly, so throttled)
YOLO_MODEL = "yolov8n.pt"
YOLO_CONF_THRESHOLD = 0.45
MAX_FACES = 3                 # >1 so a second person in frame is detectable

# --- Looking away (head pose) ---
YAW_THRESHOLD_DEG = 28              # head turn angle considered "away"
PITCH_THRESHOLD_DEG = 20
LOOK_AWAY_MIN_DURATION_SEC = 1.0    # head must stay turned this long to count once
LOOK_AWAY_WINDOW_SEC = 60           # rolling window for counting repeats
LOOK_AWAY_EVENT_COUNT = 3           # events inside the window before it alerts

# --- Looking down at notes / lap ---
# Sign convention for "down" depends on camera mounting. Watch the pitch:
# number drawn on the feed while looking down; if this check never fires,
# flip this to +1. If it fires while looking up, leave it at -1.
LOOK_DOWN_PITCH_SIGN = -1
LOOK_DOWN_PITCH_DEG = 22
LOOK_DOWN_MIN_DURATION_SEC = 2.5    # a glance at the keyboard shouldn't count

# --- Eye gaze off-screen (iris tracking, independent of head pose) ---
# Catches eyes sliding sideways to a neighbour or a second screen while the
# head stays deliberately still - the classic way head-pose-only proctors
# get beaten.
GAZE_OFFSET_THRESHOLD = 0.18        # iris deviation from centred (0 = centred)
GAZE_MIN_DURATION_SEC = 1.5
GAZE_WINDOW_SEC = 60
GAZE_EVENT_COUNT = 3

# --- Talking / lip movement ---
MAR_STD_WINDOW_SEC = 1.5            # rolling window for mouth-motion variance
MAR_STD_THRESHOLD = 0.045
TALK_WINDOW_SEC = 45
TALK_EVENT_COUNT = 4

# --- Objects in frame ---
PHONE_CONSEC_FRAMES = 2             # consecutive detections needed to confirm
NOTES_CONSEC_FRAMES = 3             # book/paper - slower, more false positives
DEVICE_CONSEC_FRAMES = 3            # laptop/monitor/keyboard/mouse/remote

# --- Second person present ---
SECOND_PERSON_CONSEC_FRAMES = 2

# --- Hand near face (earpiece, whispering, hiding the mouth) ---
HAND_NEAR_FACE_RATIO = 0.9          # distance in multiples of face width
HAND_NEAR_FACE_MIN_SEC = 2.0

# --- Student leaves the frame ---
ABSENCE_SEC = 5.0

# --- Camera tampering (covered lens, unplugged phone, frozen feed) ---
TAMPER_DARK_MEAN = 12               # frame is essentially black
TAMPER_FLAT_STD = 6                 # frame is essentially uniform
TAMPER_MIN_SEC = 3.0

# --- Alert display / rate limiting ---
ALERT_DISPLAY_SEC = 6               # how long a triggered banner stays up
ALERT_COOLDOWN_SEC = 8              # min gap between repeats of the SAME alert

# --- Logging ---
ENABLE_CSV_LOG = True
LOG_PATH = "incidents_log.csv"
SAVE_SNAPSHOTS = True               # write a JPEG next to each logged incident
SNAPSHOT_DIR = "incidents"
