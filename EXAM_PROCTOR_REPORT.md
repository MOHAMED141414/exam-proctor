# Exam Proctor Teardown

**Engineering report — single-camera proctoring**

Two reported misclassifications turned into six defects, three of them checks that had
never once fired. Every fix here came from measuring the system's own recorded output —
the guesses, where I made them, are marked as such.

| | |
|---|---|
| **Project** | exam-proctor (YOLOv8 + MediaPipe) |
| **Sessions** | 3 live runs, 17.1 min |
| **Evidence** | 273 logged incidents, 263 snapshots |

---

## Where things stand

Six defects found. Four fixed and verified, one fixed but needing a deliberate test,
one reworked but never exercised.

| Defect | Status | Evidence |
|---|---|---|
| Hand on lens read as absence | **Fixed** | 2/2 caught, 0/86 false positives |
| Notes read as device | **Fixed** | device alerts 24 → 2 |
| Look-down never fired | **Fixed** | 0 in 13 min → fires |
| Gaze-off never fired | **Needs test** | 0.18 → 0.12 → 0.07 (p90 = 0.056) |
| Left-frame unreliable | **Untested** | not exercised in any run |
| Frame rate | **Fixed** | 5.3 → 22 fps, at full resolution |

---

## Environment

Two traps here would have cost hours if hit later, and both fail silently rather than loudly.

### The dependency pin that lied

The project pins `mediapipe>=0.10.14,<1.0`, with a comment explaining that 1.x removed the
legacy `mp.solutions` API the face and hand analysis is built on. That comment is wrong in a
way that matters: **the 0.10.x line itself dropped it**. A clean install resolves to 0.10.35,
which imports fine and then fails on attribute access.

```
mediapipe 0.10.35 -> dir(mp) == ['Image','ImageFormat','tasks']
                     AttributeError: module 'mediapipe' has no attribute 'solutions'
mediapipe 0.10.14 -> mp.solutions.face_mesh  OK
                     mp.solutions.hands      OK
```

Pinned to `==0.10.14` and rewrote the comment. Installing it pulls jax, which drags numpy 2.x
in against mediapipe's own `numpy<2` constraint, so numpy is reinstalled afterwards — pip's
resulting jax warning is harmless, since nothing here uses jax.

- **Python 3.11** (uv-managed) rather than the system default 3.14 — mediapipe publishes no
  3.14 wheels.
- **TLS interception on this network** breaks pip's certificate validation while `curl`
  succeeds, because curl uses the Windows certificate store. Worked around with
  `--trusted-host`. This resurfaced later and cost two failed GPU downloads.

---

## The camera chain

Several rounds were lost to device identification. Recorded here because it is not guessable
and not stable across machines.

Windows enumerates cameras differently per capture backend, so an index means nothing without
its backend. The README's documented mapping was correct for its author's machine and wrong
for this one — it pointed "the laptop camera" at a phone virtual camera. `pygrabber` resolves
this properly by returning real device names in DirectShow order.

| Backend / index | Device | State |
|---|---|---|
| `MSMF 0` | DroidCam Video | **Decoy** — permanent placeholder |
| `DSHOW 1` | HP Wide Vision HD Camera | the built-in laptop webcam |
| `DSHOW 2` | S22 Ultra (Windows Virtual Camera) | phone, via Windows |
| `DSHOW 3` | OBS Virtual Camera | **In use** — the phone feed |

The phone does *not* arrive through the DroidCam virtual camera. The DroidCam OBS source
publishes into an OBS scene, and OBS's own virtual camera is what other applications can open:

```
phone DroidCam app -> DroidCam OBS source -> OBS scene
                   -> Start Virtual Camera -> DSHOW 3
```

Direct HTTP streaming (`http://ip:4747/video`) does not work on this DroidCam build — the port
accepts TCP but serves no MJPEG, so the client handshake is required. If OBS's virtual camera
is stopped, the feed reverts to a placeholder and the app correctly reports `CAMERA BLOCKED`.

---

## Fault 1 — a hand on the lens read as "student left the frame"

**Status: fixed.** The first reported symptom, diagnosed from the two real occlusion frames
the system had already saved.

Tamper detection tested two things: is the frame *dark* (`mean<12`), is it *flat* (`std<6`).
A hand pressed to a lens is neither — it is bright, and lit unevenly enough to carry real
variance. So the check could never fire, every downstream detector reported "nobody there",
and the absence rule claimed the event instead.

**Measured — occluded vs normal frames:**

```
                        mean     std   lapVar
hand over lens          73.3    62.9      8.7
hand over lens         129.5    20.9     17.7
                     -------  ------   ------
normal (5 frames)    150-163   57-70   106-191

thresholds: dark < 12   flat < 6    <- neither separates these
focus:      8.7-17.7  vs  106-191   <- ~6x separation
```

Brightness and variance overlap completely; **focus** separates them cleanly. Threshold set to
45 — the geometric midpoint of 17.7 and 106.3.

### What changed

- Added a **Laplacian-variance blur test** on a half-size frame — it runs every frame, unlike
  the throttled detectors. Measured cost: `0.9 ms`.
- Added the **frozen-feed test** the method's own docstring already advertised but which was
  never implemented.
- Fixed a latent bug this exposed: `last_person_seen` went stale while blocked, so uncovering
  the lens after a long occlusion could fire a bogus absence alert.
- The alert now records *why*: `feed dark`, `feed flat`, `lens covered or defocused`,
  `feed frozen`.

**Verified offline against all 88 snapshots:**

```
must be BLOCKED  ->  2/2 caught  (lapVar 8.7, 17.7)
must NOT block   ->  0/86 false positives
RESULT: PASS
```

---

## Fault 2 — notes reported as a device

**Status: fixed.** The second reported symptom. The diagnosis reframed it entirely: it was
barely about notes.

**Measured — every detection across 88 snapshots, conf ≥ 0.20:**

```
tv           59 / 88 frames   the classroom wall monitor, pinned to the right edge
laptop       16               background objects + weak notes misreads
cell phone    5
book          1  @ conf 0.21  <- below the 0.45 gate: notes detection was dead
remote        1
keyboard      1
```

The 24 device alerts were spaced 8–9 s apart — exactly `ALERT_COOLDOWN_SEC`. That is the
signature of a permanently-true condition re-firing at the cooldown floor, not of intermittent
events.

Two independent faults compounded into one symptom: **static room furniture** held the device
banner up continuously, while `book` never fired to label the notes correctly. Opening notes
appeared to trigger "device" because that alert was already on.

### What changed

- **yolov8n → yolov8s.** The decisive difference: notes went from a single 0.21 detection to
  `0.52 / 0.63 / 0.79` — above even the old global threshold.
- **Per-category confidence floors** replacing one global gate. The categories are not equally
  hard: `notes` at 0.25 (catch more, tolerate false alarms, as chosen), `device` at 0.55 to
  hold back weak background hits.
- **A static-background filter** learning whatever holds still during the first 10 s and
  ignoring it thereafter. `person` is exempt unconditionally — suppressing a motionless second
  person would convert a detection into a blind spot.
- **Dropped `tv` from the device category entirely.** This is the change that actually worked.

> **Where I was wrong.** The background filter was the plan's centrepiece and it
> **underperformed badly**. It first learned nothing at all (the ratio gate was too strict for
> flaky detection); after tuning it suppressed 27%. The cause was structural: it learns fixed
> positions, which assumes a stationary camera — but a laptop webcam shifts every time the lid
> is adjusted, and `tv` was being detected at four distinct frame positions. Calibrating on a
> contiguous stable window instead reached 61%, confirming the diagnosis. Dropping `tv` is what
> took device alerts from 18 to 2. I also introduced and then fixed a reporting bug in it:
> `summary()` claimed nothing had been learned while the filter was actively suppressing,
> because the lazy build ran only inside `filter()`.

---

## Three checks that had never fired

Reported after the second run. Confirmed against both session summaries: look-down and gaze-off
had fired **zero** times in 13 minutes of running.

### Look-down — a threshold ordering bug (fixed)

```
PITCH_THRESHOLD_DEG  = 20    look-away fires when |pitch| > 20
LOOK_DOWN_PITCH_DEG  = 22    look-down fires when  pitch  < -22

-> every downward tilt satisfied look-away BEFORE reaching look-down
-> look-away fired 31 times across two sessions; look-down fired 0
```

Lowered to 14, *and* stopped look-away counting downward tilt at all, so the two no longer
compete for the same head position. Turning aside and looking up still register as look-away.
Verified in isolation — look-down 1, look-away 0 for the same pose — then confirmed live.

### Gaze-off — a threshold outside the signal range (needs a deliberate test)

After this failed a third session, I stopped adjusting it by reasoning and sampled the actual
metric from the session's own snapshots.

**Measured — 36 frames containing a face:**

```
gaze_offset   min 0.006   median 0.027   p90 0.056   max 0.123

threshold 0.12  -> exceeded in 1 of 36 frames
                -> then required sustained 1.0 s, twice in 60 s

head_forward (|yaw| < 28)  ->  32/36 = 89% pass
```

The threshold sat at the very top of the range this metric reaches on this geometry; the
original 0.18 was beyond it entirely. Set to **0.07**, just above p90. This also *disproved*
my stated hypothesis that the `head_forward` gate was suppressing the check — 89% of frames
pass it.

### Left-frame — reworked, still unverified (untested)

- Kept **deliberately separate** from `CAMERA BLOCKED`, as specified: a covered lens and an
  empty chair are different events, even though both leave the proctor unable to see anyone.
- The absence timer only advances while the feed passes *every* tamper check — camera
  verifiably working.
- Now also requires **3 consecutive empty detection sweeps**, so a momentary detector dropout
  cannot read as an empty room.
- **Not exercised in any run.** It logged 0, but no run confirmed the student actually left
  frame — so this is untested, not working.

---

## Performance — 5.3 → 22 fps

Profiled before changing anything, which redirected the work twice.

| Component | Cost | Runs | Per frame | Share |
|---|---|---|---|---|
| YOLO yolov8s @ 640 (CPU) | 464 ms | every 3rd | 155 ms | **82%** |
| MediaPipe FaceMesh + iris | 18.8 ms | every frame | 18.8 ms | 10% |
| MediaPipe Hands | 42.8 ms | every 3rd | 14.3 ms | 8% |
| Blur test (added this session) | 0.9 ms | every frame | 0.9 ms | <1% |

Two findings changed the plan: the blur check I had added was **not** a meaningful cost, and
reducing `MAX_FACES` from 3 to 2 produced **no gain** (18.8 vs 20.6 ms — noise), which killed
an obvious-looking lever. Everything hinged on YOLO.

> **The catch worth remembering.** The machine has an **NVIDIA GTX 1050**, but Torch was the
> CPU-only build, so the GPU sat idle. After installing CUDA Torch,
> `torch.cuda.is_available()` returned `True` — and ultralytics **still ran on the CPU**,
> reporting `device: cpu`. CUDA being available does not mean the model uses it. That was a
> silent 6× loss the availability check would have hidden; the device is now pinned explicitly.

| Configuration | Sweep | Overall | Accuracy cost |
|---|---|---|---|
| CPU @ 640, every 3rd (original) | 464 ms | 5.3 fps | — |
| CPU @ 320, every 5th | 199 ms | 13.6 fps | phone detection degraded |
| **GPU @ 640, every 3rd (final)** | 34.6 ms | **22 fps** | none — full resolution |

The GPU allowed both earlier compromises to be reverted: the final configuration is faster
*and* more accurate than any previous state. Pascal (`sm_61`) is supported in the cu126 build,
so no fallback was needed. **The bottleneck has moved** — YOLO is now ~11 ms/frame while
FaceMesh and Hands are ~73% of the budget. Further speed work belongs there, not in YOLO.

The install took three attempts, each failing differently and *each reporting exit code 0 while
failing*: the redirect host `download-r2.pytorch.org` was missing from the TLS bypass list;
then pip's 15-second read timeout aborted a 2.5 GB transfer; then success with `--timeout 300`.

---

## People counting

**Status: delivered.** Requested change: report how many people are present, not merely that a
second person exists.

- The check collapsed everything to `max(bodies, faces) > 1`. It now reports the count:
  `4 people in frame (bodies=4 faces=3)`.
- `MAX_FACES` raised **3 → 6** — at 3, a room of six silently read as three. Profiling had
  already shown cost is flat across this range.
- Bodies and faces are counted separately and the larger wins: YOLO merges adjacent people into
  one box at close range where FaceMesh still resolves both, and someone turned away has a body
  but no face.
- Live `people N` row on the panel (amber when >1), plus `Most people in frame at once` in the
  session summary. Confirmed at **4** in the live run.

---

## Session-over-session

Per-minute rates, since the sessions differ in length. Sessions 1–2 used the laptop webcam;
session 3 used the phone via OBS, on GPU.

| Alert | S1 /min | S2 /min | S3 /min | Reading |
|---|---|---|---|---|
| Camera blocked | 0.00 | 0.35 | 0.00 | fault 1 fixed |
| Student left frame | 0.28 | 0.00 | 0.00 | no longer misfires; untested |
| Notes / book visible | 0.00 | 1.23 | 0.24 | detection now possible |
| Extra device visible | 3.33 | 3.16 | **0.48** | dropping `tv` worked |
| Looking down (notes) | 0.00 | 0.00 | 0.24 | fires at last |
| Gaze off screen | 0.00 | 0.00 | 0.00 | still unproven |
| Phone / device visible | 0.42 | 1.05 | **3.33** | needs review — see below |
| Second person present | 1.11 | 2.46 | 6.43 | genuine — 4 people present |
| Talking detected | 3.75 | 4.21 | 4.52 | never tuned |

> **A regression I caused, and its correction.** Between S1 and S2, total incidents rose from
> **12.2 to 17.2 per minute**. Upgrading to yolov8s fixed notes detection but made the model
> better at detecting *everything*, and my background filter did not offset it. That regression
> is what forced the `tv` decision, which brought device alerts down to 0.48/min in S3. It is
> reported here because a summary showing only the fixed checks would have been misleading.

---

## Open items

- **Gaze-off is unproven.** The threshold is now fitted to measured data, but no run has
  included a deliberate sideways glance. It has failed three sessions; assume nothing until it
  fires.
- **Left-frame is untested.** Reworked but never exercised.
- **Phone alerts rose to 3.33/min** in S3 — expected, since full 640 resolution on GPU improves
  small-object detection. But three other people are in frame and some detections are likely
  *their* phones. Review snapshots before trusting this count.
- **Talking (4.52/min) has never been tuned** and is the largest remaining alert source after
  second-person.
- **The background filter is weak for moving cameras.** Less critical with a phone on a stand,
  but it still learns fixed positions once and never re-learns.
- **Thresholds are fitted to one room and one camera.** `TAMPER_BLUR_LAPVAR=45` especially —
  dimmer light lowers Laplacian variance across the board, so a threshold set in a bright room
  will read a dim one as blocked.
- **This remains a prototype.** The README's own caveat stands: expect false positives, and get
  institutional sign-off before recording students — snapshots are personal data, and the app
  handles neither consent nor retention.

---

## Changed files

| File | Change |
|---|---|
| `config.py` | Camera source; blur + frozen thresholds; background calibration; per-category confidence; gaze, look-down and absence tuning; `MAX_FACES`; imgsz |
| `alert_manager.py` | Blur/frozen tamper detection; look-away vs look-down separation; absence confirmation; people counting |
| `object_detector.py` | Per-category gating; `StaticBackgroundFilter` + `iou()`; explicit CUDA device; `tv` removed |
| `main.py` | Background filter wiring; calibration state; `down:Y/N` live indicator; people row |
| `requirements.txt` | mediapipe pinned to `==0.10.14` with corrected rationale |
| `README.md` | Check 11 criteria; calibration and tuning notes; performance section |
| `.gitignore` | Widened to `incidents*/` — renamed baseline snapshots are personal data and fell outside the old patterns |

The first session's evidence was preserved rather than deleted, as
`incidents_baseline_before_fix/` — those snapshots are photographs of a person, and the
deletion would have been irreversible.

---

*Every threshold in this report is fitted to one camera in one room. Re-measure rather than
reason about them if the setup changes — that method found four of the six defects, while the
two hypotheses I reasoned out instead were both wrong.*
