"""
Central configuration for the exam proctoring system.

Camera: currently set to the laptop's built-in webcam for a local trial run.
Edit these values to match your setup before running main.py.
"""

# --- Camera source ---
# Windows enumerates cameras differently per capture backend, so the backend
# is pinned rather than left to OpenCV. MSMF and DSHOW do NOT agree on index
# order on this machine (MSMF index 1 turned out to be a phone virtual
# camera, not the laptop). DSHOW's order was confirmed authoritatively via
# pygrabber's device list, which names each index directly:
#     DSHOW index 0 = DroidCam Video   (does not open under DSHOW - use MSMF)
#     DSHOW index 1 = HP Wide Vision HD Camera   (the HP OMEN's own camera)
#     DSHOW index 2 = phone virtual camera (Windows Virtual Camera)
#     DSHOW index 3 = OBS Virtual Camera
#     MSMF  index 0 = DroidCam         <-- the one we want
# Leaving the backend unset makes the index ambiguous between devices.
#
# DroidCam must actually be streaming: start the phone app and connect the
# DroidCam client first. Until then the feed is the "Start DroidCam" placeholder
# (brightness ~4), which the proctor correctly reports as CAMERA BLOCKED.
# The phone arrives via OBS, not the DroidCam virtual camera: the DroidCam OBS
# source publishes into an OBS scene, and OBS's own virtual camera is what any
# other app can actually open. MSMF 0 ("DroidCam Video") stays on the "Start
# DroidCam" placeholder the whole time and is NOT the right device here.
# OBS must be running with Start Virtual Camera pressed, or this reads a
# placeholder and every check reports CAMERA BLOCKED.
CAMERA_SOURCE = 3
CAMERA_BACKEND = "DSHOW"
CAMERA_NAME = "Phone via OBS Virtual Camera"

# Alternative: skip the DroidCam Client and pull the stream over WiFi using
# the IP:port the DroidCam phone app shows. Set CAMERA_SOURCE to the URL,
# e.g. "http://192.168.1.50:4747/video" - the backend setting is then ignored.

# Position the phone so it sees the student face AND the desk surface.
# Every check runs on this one feed, so framing matters more than it did with
# two cameras: a face-only crop gives up notes- and phone-on-desk detection.

# --- Seats (multi-student mode) ---
# The frame is split into SEAT_COUNT equal vertical strips and every face,
# hand and object is attributed to the strip its centre falls in. Each seat
# then runs the full check set against its own independent state, so one
# student looking away does not consume another's rolling window.
#
# The whole system was originally written around a single student, where a
# second face meant cheating. Here extra faces are expected, and the violation
# becomes TWO FACES IN ONE SEAT - somebody leaning into a neighbour's space.
#
# Fixed strips assume people stay roughly in their own third. Leaning across a
# boundary misattributes that frame; seats are assigned per frame, not tracked.
SEAT_COUNT = 3
SEAT_LABELS = ["LEFT", "MID", "RIGHT"]

# --- Capture resolution ---
# Three faces in one frame makes each face roughly a third the width it would
# be alone. At 640x480 that leaves ~100px per face, which is below what the
# iris landmarks behind the gaze check need. 720p roughly doubles it.
# Costs MediaPipe time (it scales with input), so lower this first if the
# frame rate suffers.
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080

# --- Display ---
DISPLAY_WIDTH = 860
DISPLAY_HEIGHT = 600
PANEL_WIDTH = 400            # right-hand live checklist: one column per seat
WINDOW_NAME = "Exam Proctor - press 'q' to quit"

# --- Performance ---
# Tuned against a CUDA GPU (GTX 1050). Measured sweep cost for yolov8s at 640:
# 222ms on CPU against 34ms on GPU, so YOLO stopped being the bottleneck - the
# per-frame budget is now mostly FaceMesh (19ms) and Hands (43ms every 3rd).
# That buys back full inference resolution AND the faster cadence at ~22 fps.
#
# On a CPU-only machine these settings give roughly 5 fps. Drop YOLO_IMGSZ to
# 320 and raise DETECT_EVERY_N_FRAMES to 5 there (~14 fps), at the cost of
# small-object detection - a phone is the first thing missed at 320.
DETECT_EVERY_N_FRAMES = 3     # YOLO cadence
HANDS_EVERY_N_FRAMES = 3      # MediaPipe Hands cadence (costly, so throttled)
YOLO_MODEL = "yolov8s.pt"     # 'n' confused open notebooks with laptops too often
YOLO_IMGSZ = 640              # full resolution; see note above before lowering
# Caps how many people can be counted, so it must exceed the largest crowd you
# want reported - at 3 a room of six still reads as three. Measured cost is flat
# across this range (max_faces 2 and 3 both timed ~19ms), so the ceiling is
# cheap; raise it further if you need to count bigger rooms.
MAX_FACES = 6
# Detect faces inside each seat's crop rather than across the whole frame.
# Needed when students sit far enough away that a face is a small fraction of
# the frame - MediaPipe downscales to ~192px internally, so a 50px face in a
# 1280px frame is gone before detection starts. Costs one face-detection pass
# per seat instead of one per frame. Turn off if the camera is close enough.
ZONE_CROP_FACES = True

# Fraction of frame height searched for faces, measured from the top. The rest
# is assumed to be desk surface. This is not a nicety: with the phone lying low
# the bottom 38% of the frame was bare tabletop, and including it pushed faces
# below MediaPipe's detection floor entirely - measured on a real frame, the
# same crop found 0 faces with the tabletop and 1 without it.
# Raise toward 1.0 if the camera is mounted high and sees no desk.
ZONE_CROP_BOTTOM = 0.62
# Two hands per seat. This is the most expensive check in the system, so it is
# the first thing to cut if three seats prove too slow.
MAX_HANDS = 6

# Base floor handed to YOLO. Deliberately low: the real gate is the per-category
# table below, which lets 'notes' through at a confidence that would flood every
# other class. Anything under this never reaches the category filter at all.
YOLO_CONF_THRESHOLD = 0.20

# Minimum confidence per coarse category. Measured over a 7-minute session on
# this camera: a background wall monitor scored 'tv' at 0.42-0.69, while held-up
# paper only ever reached 'book' at 0.21 - hence the low notes floor and the
# high device floor. Notes trade precision for recall on purpose: a false notes
# alert costs a glance at the snapshot, a missed one costs the whole check.
CATEGORY_CONF_MIN = {
    "person": 0.50,
    "phone": 0.45,
    "notes": 0.25,
    "device": 0.55,
}

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
# Must stay BELOW PITCH_THRESHOLD_DEG above. When it sat higher (22 vs 20) every
# downward tilt satisfied "looking away" before it could ever reach "looking
# down", so this check never fired once in 13 minutes of logged running while
# look-away fired 31 times. update_head_pose now also stops counting downward
# tilt as look-away, so the two no longer compete for the same head position.
LOOK_DOWN_PITCH_DEG = 14
LOOK_DOWN_MIN_DURATION_SEC = 2.0    # a glance at the keyboard shouldn't count

# --- Eye gaze off-screen (iris tracking, independent of head pose) ---
# Catches eyes sliding sideways to a neighbour or a second screen while the
# head stays deliberately still - the classic way head-pose-only proctors
# get beaten.
# Fitted to measured data, after this check failed to fire in three sessions.
# Sampled over 36 frames on this camera the iris offset ran: median 0.027,
# p90 0.056, max 0.123. The original 0.18 - and then 0.12 - sat at or beyond
# the top of the range the metric actually reaches here, so no amount of
# looking sideways could trip it. 0.07 sits just above p90.
#
# Note this is a ratio of iris travel within the eye opening, so it is bounded
# well below the 0.5 its formula suggests: seating distance and camera angle
# cap it. Re-measure rather than reason about it if the camera moves.
GAZE_OFFSET_THRESHOLD = 0.07        # iris deviation from centred (0 = centred)

# Iris tracking needs the eye to span real pixels. Below this face width the
# landmarks are noise - a 41px face measured 0.419 where the true range is
# 0.006-0.123 - so the check falls back to head direction instead.
GAZE_MIN_FACE_PX = 110

# Fallback rule: a sustained MODERATE turn, below the look-away threshold.
# Looking at a neighbour's desk is a steady partial turn, not the repeated
# turn-and-return that look-away counts, and at 15-28 degrees it currently
# falls between the two checks and is missed entirely.
#
# Be clear about what this gives up: head pose cannot catch a student who
# holds their head still and moves only their eyes. That is the exact attack
# the iris check existed to stop, and at classroom range it is not detectable.
# Accepting that is the price of monitoring several students from one camera.
GAZE_HEAD_YAW_DEG = 15
GAZE_MIN_DURATION_SEC = 1.0
GAZE_WINDOW_SEC = 60
GAZE_EVENT_COUNT = 2

# --- Talking / lip movement ---
MAR_STD_WINDOW_SEC = 1.5            # rolling window for mouth-motion variance
MAR_STD_THRESHOLD = 0.045
TALK_WINDOW_SEC = 45
TALK_EVENT_COUNT = 4

# --- Objects in frame ---
PHONE_CONSEC_FRAMES = 2             # consecutive detections needed to confirm
NOTES_CONSEC_FRAMES = 4             # book/paper - persistence offsets the low conf floor
DEVICE_CONSEC_FRAMES = 3            # laptop/monitor/keyboard/mouse/remote

# --- Static background suppression ---
# Room furniture is not a cheating aid. A wall monitor behind the student fired
# 'device' in 59 of 88 logged frames, re-alerting at the cooldown floor for the
# whole session. Anything sitting still in the same spot through the opening
# window is learned once and ignored thereafter.
#
# The tradeoff is explicit: a device that is present and motionless from the
# very first second gets learned as furniture too. 'person' is never filtered -
# a second person sitting still is exactly what check 8 exists to catch.
BACKGROUND_CALIBRATION_SEC = 10.0   # opening window used to learn the room
BACKGROUND_IOU_THRESHOLD = 0.5      # overlap counting as "same object, same place"
# Deliberately below half: YOLO drops a given background object in a good third
# of frames, so requiring it in most of them learns nothing. Measured on the
# logged session the wall monitor held one position in 9 of 20 sweeps (0.45),
# which is emphatically furniture. Position stability is the real signal here,
# and the IoU threshold above is what enforces it.
BACKGROUND_MIN_SEEN_RATIO = 0.4     # share of calibration sweeps it must appear in

# --- Second person present ---
SECOND_PERSON_CONSEC_FRAMES = 2

# --- Hand near face (earpiece, whispering, hiding the mouth) ---
HAND_NEAR_FACE_RATIO = 0.9          # distance in multiples of face width
HAND_NEAR_FACE_MIN_SEC = 2.0

# --- Student leaves the frame ---
# This means one specific thing: the camera is working and the student is not
# in front of it. It is deliberately NOT merged with CAMERA BLOCKED - a covered
# lens and an empty chair are different events with different responses, even
# though both leave the proctor unable to see anyone. The absence timer only
# advances while the feed passes every tamper check.
ABSENCE_SEC = 5.0
ABSENCE_CONSEC_SWEEPS = 3           # detection sweeps with nobody found, in a row

# --- Camera tampering (covered lens, unplugged phone, frozen feed) ---
TAMPER_DARK_MEAN = 12               # frame is essentially black
TAMPER_FLAT_STD = 6                 # frame is essentially uniform
TAMPER_MIN_SEC = 3.0

# A hand over the lens is neither dark nor flat - it is bright and textured, so
# the two thresholds above cannot see it. Measured on this camera: hand-over-lens
# frames scored mean 73/130 and std 63/21 (indistinguishable from a normal
# frame), but Laplacian variance 8.7 and 17.7 against 106-191 for real frames.
# Focus is the signal. 45 sits at the geometric midpoint of those two ranges.
# Re-measure if the room lighting changes - dim light lowers the whole scale.
# Re-measured on the phone at 1080p across a whole classroom, where legitimate
# frames ran min 35.7 / median 255 / max 550. The old 45 - fitted to a laptop
# webcam at 640x480 - clipped the bottom of that range and raised 16 false
# CAMERA BLOCKED alerts in one session with nothing ever covering the lens.
#
# This value does not transfer between cameras, resolutions or rooms. A wider
# scene, a softer lens or dimmer light all move the whole scale. Re-measure it
# rather than reasoning about it whenever the setup changes.
TAMPER_BLUR_LAPVAR = 20             # below this the lens is covered or defocused
TAMPER_FROZEN_DIFF = 0.5            # mean abs frame-to-frame delta = dead feed

# --- Alert display / rate limiting ---
ALERT_DISPLAY_SEC = 6               # how long a triggered banner stays up
ALERT_COOLDOWN_SEC = 8              # min gap between repeats of the SAME alert

# --- Logging ---
ENABLE_CSV_LOG = True
LOG_PATH = "incidents_log.csv"
SAVE_SNAPSHOTS = True               # write a JPEG next to each logged incident
SNAPSHOT_DIR = "incidents"
