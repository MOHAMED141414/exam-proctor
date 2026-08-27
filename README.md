# Exam Proctor - single-camera cheating detection

Watches a student through **one camera: an iPhone running DroidCam**, and
raises real-time on-screen alerts for eleven behaviours. The laptop webcam is
deliberately not used.

| # | Alert | What trips it |
|---|---|---|
| 1 | Looking away repeatedly | Head yaw/pitch past threshold, 3 times in 60s |
| 2 | Looking down (notes) | Sustained downward head tilt for 2.5s |
| 3 | Gaze off screen | Irises held off-centre while the head stays forward |
| 4 | Talking detected | Repeated mouth movement in a rolling window |
| 5 | Phone / device visible | YOLO `cell phone` |
| 6 | Notes / book visible | YOLO `book` |
| 7 | Extra device visible | YOLO `laptop`, `tv`, `keyboard`, `mouse`, `remote` |
| 8 | Second person present | More than one body or more than one face |
| 9 | Hand at ear / mouth | Hand landmarks held near an ear or the lips |
| 10 | Student left frame | No person and no face for 5 continuous seconds |
| 11 | Camera blocked | Feed goes dark, flat, out of focus, or frozen for 3s |

Checks 2, 3, 7, 8, 9 and 11 are the additions beyond the original four.

## 1. Install

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

A virtual environment is strongly recommended: mediapipe pulls in a
protobuf version that conflicts with TensorFlow if one is installed
globally, and mediapipe fails to import when that happens.

The version ceilings in `requirements.txt` are load-bearing, not caution:
mediapipe 1.x removed the `mp.solutions` API this code is built on, and
OpenCV 5 requires numpy>=2 while mediapipe 0.10.x pins numpy<2.

YOLOv8n weights download automatically on first run (needs internet once).

## 2. Set up the iPhone as the camera

1. Install **DroidCam** from the App Store and the **DroidCam Client** on
   the PC (dev47apps.com). Connect both to the same WiFi, then connect the
   client to the phone. Until you do, the feed shows a "Start DroidCam"
   placeholder and the proctor correctly reports `CAMERA BLOCKED`.
2. Confirm which index the phone is on. **Windows numbers cameras
   differently per capture backend**, so `CAMERA_BACKEND` is pinned in
   `config.py` and the index only means anything alongside it. As verified
   on this machine:

   | Backend | index 0 | index 1 | index 2 |
   |---|---|---|---|
   | MSMF (default) | laptop "HD camera" | **DroidCam** | - |
   | DSHOW | - | laptop | OBS Virtual Camera |

   If your numbering differs, run a short probe over indices 0-3 on each
   backend and look at what comes back before trusting an index.
3. No client, WiFi only: set `CAMERA_SOURCE` to the URL the phone app shows,
   e.g. `"http://192.168.1.50:4747/video"`. The backend setting is then
   ignored.

### Camera placement matters more than it used to

With one camera, framing is a real trade-off: every check runs on this single
feed. Angle the phone so it sees **both the face and the desk surface**. A
tight face crop keeps checks 1-4 working but gives up notes and
phone-on-desk detection entirely.

## 3. Run it

```bash
.venv\Scripts\python.exe main.py
```

The window shows the feed plus a live checklist panel: green means never
tripped, amber means tripped earlier, red means firing right now, with a
running count per check. Press `q` to quit and a session summary prints.

Every alert is written to `incidents_log.csv` with a timestamp, and a JPEG of
the moment is saved to `incidents/` as evidence (`SAVE_SNAPSHOTS`).

## How the checks are structured

Four rule shapes do all the work, chosen per behaviour to suppress noise:

- **repeat-in-window** - N events inside a rolling window. Used where a single
  occurrence is innocent: one glance away is nothing, three in a minute is not.
- **sustained** - the condition holds continuously for N seconds. Used where
  duration is the signal: reading notes means a steady downward tilt, not the
  turn-and-return of a glance.
- **consecutive** - N detections in a row, filtering single-frame YOLO blips.
- **timeout** - nothing seen for N seconds (left frame).

Every alert then passes a per-label cooldown (`ALERT_COOLDOWN_SEC`), so a
condition that stays true cannot machine-gun the log.

Two design notes worth knowing:

- **Gaze is checked independently of head pose.** Head-pose-only proctoring is
  beaten by holding the head still and moving just the eyes; the iris check
  covers that. It is suppressed at steep yaw, where iris geometry is
  foreshortened and the head-pose check already owns the case.
- **Camera tampering is checked first, and suppresses the absence alert.** On
  a dead feed every detector reports "nothing there", which is otherwise
  indistinguishable from the student having walked away.
- **Blocking is judged on focus, not just brightness.** A hand over the lens is
  bright and textured, so dark/flat tests miss it entirely and the student reads
  as absent. Measured on one camera: occluded frames scored 8.7-17.7 Laplacian
  variance against 106-191 for normal ones, so focus separates them cleanly.
- **The room is learned before it is judged.** Furniture is not a cheating aid,
  but YOLO cannot tell a wall monitor from a smuggled one. For the first
  `BACKGROUND_CALIBRATION_SEC` the proctor watches without alerting and records
  whatever holds still; those positions are ignored for the rest of the session.
  A device present and motionless from the very first second is learned as
  furniture too - that is the price of the check. `person` is never learned.

## Tuning

All thresholds live in `config.py`. The feed prints live `yaw:`, `pitch:`,
`gaze:` and `faces:` values - watch them and set thresholds to match your
actual camera angle and seating distance.

`LOOK_DOWN_PITCH_SIGN` deserves a specific mention: which sign means "down"
depends on how the phone is mounted. Watch the `pitch:` number while looking
down at the desk. If check 2 never fires, flip the sign to `+1`; if it fires
while you look up, leave it at `-1`.

`TAMPER_BLUR_LAPVAR` is fitted to one camera in one room and is the knob to
re-measure first if check 11 misbehaves - dimmer light lowers Laplacian
variance across the board, so a threshold set in a bright room will read a dim
one as blocked. `CATEGORY_CONF_MIN` sets the confidence bar per category rather
than globally, because the categories are not equally hard: the `notes` floor
is low on purpose (catch more, tolerate false alarms) and `device` is high to
hold back weak background hits.

## Performance

- `DETECT_EVERY_N_FRAMES` controls YOLO cadence, `HANDS_EVERY_N_FRAMES`
  controls hand tracking (the most expensive check). Raise either on a slower
  machine - watch the `fps` readout in the panel and raise the cadence if it
  drops below roughly 8.
- `yolov8s.pt` is the default. `yolov8n.pt` is faster but confuses an open
  notebook with a laptop badly enough to make check 6 useless: on one logged
  session `n` detected held-up notes once in 88 frames, at 0.21 confidence,
  where `s` found them at 0.52-0.79.
- ultralytics auto-uses a CUDA GPU if present, otherwise CPU.

## Known limitations

- A working prototype, not a validated proctoring product. Expect false
  positives and tune before relying on it for real grading decisions.
- Detection quality depends heavily on lighting. A backlit subject (window
  behind the student) washes out the face and MediaPipe will find nothing -
  put the light source in front of the student, not behind.
- `notes` and `device` lean on COCO `book`/`laptop`/`tv`, which are imperfect
  proxies: loose paper is often missed, and a book cover can read as a laptop.
  Treat checks 6 and 7 as prompts to review the snapshot, not as proof.
- Head pose from a 6-point solvePnP is approximate and depends on camera
  angle; recalibrate per room and seat.
- Assumes one student per station. No identity verification is included, so
  the system cannot tell whether the right person is sitting the exam.
- The second-person check needs the helper to be visible in frame; someone
  off-camera feeding answers is only caught indirectly, via checks 3, 4 or 9.
- Before deploying this in an actual exam, check your institution's policy on
  recording students and get sign-off. Snapshots are on by default and are
  personal data. This script does not handle consent or retention rules.
