import time

import cv2
import numpy as np

import config
from alert_manager import ALERT_ORDER, CAMERA_BLOCKED, AlertManager
from camera_stream import VideoStream
from face_analyzer import FaceAnalyzer, HandAnalyzer, hand_near_face
from object_detector import (ObjectDetector, StaticBackgroundFilter,
                             count_label, has_label)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Checks that belong to a student. CAMERA_BLOCKED is a property of the feed
# itself, not of any one seat, so it is tracked separately and shown once.
SEAT_CHECKS = [a for a in ALERT_ORDER if a != CAMERA_BLOCKED]

SEAT_TINTS = [(120, 200, 120), (200, 180, 90), (190, 130, 200)]


def seat_of(x, width, seat_count):
    """Which vertical strip a horizontal position falls in.

    Seats are decided per frame from position alone - there is no tracking, so
    leaning across a boundary hands that frame to the neighbouring seat.
    """
    if width <= 0:
        return 0
    idx = int(x // (width / seat_count))
    return max(0, min(seat_count - 1, idx))


def draw_seat_zones(frame, seat_count, labels):
    h, w = frame.shape[:2]
    for i in range(seat_count):
        x = int(w * i / seat_count)
        if i:
            cv2.line(frame, (x, 0), (x, h), (90, 90, 90), 1)
        cv2.putText(frame, labels[i], (x + 10, h - 12), FONT, 0.5,
                    SEAT_TINTS[i % len(SEAT_TINTS)], 1)


def draw_faces(frame, seated_faces, labels):
    """Each face gets its own readout, drawn at its own box."""
    for seat, face in seated_faces.items():
        tint = SEAT_TINTS[seat % len(SEAT_TINTS)]
        x1, y1, x2, y2 = face.face_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), tint, 1)

        down = (face.pitch * config.LOOK_DOWN_PITCH_SIGN) > config.LOOK_DOWN_PITCH_DEG
        cv2.putText(frame, f"{labels[seat]} y{face.yaw:.0f} p{face.pitch:.0f}",
                    (x1, max(14, y1 - 20)), FONT, 0.45, tint, 1)
        cv2.putText(frame, f"g{face.gaze_offset:.2f} down:{'Y' if down else 'N'}",
                    (x1, max(28, y1 - 6)), FONT, 0.45,
                    (0, 255, 255) if down else tint, 1)


def draw_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), d.color, 2)
        cv2.putText(frame, f"{d.label} {d.conf:.2f}", (x1, max(12, y1 - 8)),
                    FONT, 0.5, d.color, 2)


def draw_banner(frame, text):
    w = frame.shape[1]
    if text:
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 200), -1)
        shown = text if len(text) < 66 else text[:63] + "..."
        cv2.putText(frame, f"ALERT: {shown}", (10, 27), FONT, 0.6, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (0, 0), (w, 30), (0, 120, 0), -1)
        cv2.putText(frame, "Monitoring...", (10, 21), FONT, 0.55, (255, 255, 255), 1)


def draw_panel(managers, camera_mgr, labels, height, width, fps, elapsed,
               calibrating_sec):
    """One row per check, one column per seat. A single banner cannot show
    three students at once, so the grid is what you actually read."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (28, 28, 28)

    n = len(managers)
    col_w = 76
    x0 = width - col_w * n - 10

    cv2.putText(panel, "CHECKS", (12, 24), FONT, 0.55, (255, 255, 255), 2)
    for i, label in enumerate(labels):
        cv2.putText(panel, label, (x0 + i * col_w + 6, 24), FONT, 0.45,
                    SEAT_TINTS[i % len(SEAT_TINTS)], 1)
    cv2.line(panel, (12, 34), (width - 12, 34), (70, 70, 70), 1)

    statuses = [dict((lbl, (act, cnt)) for lbl, act, cnt in m.get_status())
                for m in managers]

    y = 58
    for check in SEAT_CHECKS:
        name = check if len(check) <= 20 else check[:19] + "."
        cv2.putText(panel, name.title(), (12, y), FONT, 0.4, (215, 215, 215), 1)
        for i, st in enumerate(statuses):
            active, count = st[check]
            color = (60, 60, 255) if active else ((90, 190, 90) if count == 0
                                                  else (0, 165, 255))
            cx = x0 + i * col_w + 12
            cv2.circle(panel, (cx, y - 5), 5, color, -1)
            cv2.putText(panel, str(count), (cx + 14, y), FONT, 0.42, color, 1)
        y += 24

    y += 6
    cv2.line(panel, (12, y), (width - 12, y), (70, 70, 70), 1)
    y += 22

    cam_active, cam_count = next(
        (a, c) for lbl, a, c in camera_mgr.get_status() if lbl == CAMERA_BLOCKED)
    cam_color = (60, 60, 255) if cam_active else ((90, 190, 90) if cam_count == 0
                                                  else (0, 165, 255))
    cv2.circle(panel, (20, y - 5), 5, cam_color, -1)
    cv2.putText(panel, f"Camera blocked   {cam_count}", (34, y), FONT, 0.42,
                (215, 215, 215), 1)
    y += 24

    cv2.putText(panel, "people in seat", (12, y), FONT, 0.4, (150, 150, 150), 1)
    for i, m in enumerate(managers):
        c = (0, 165, 255) if m.people_now > 1 else (180, 180, 180)
        cv2.putText(panel, str(m.people_now), (x0 + i * col_w + 12, y), FONT, 0.42, c, 1)
    y += 26

    cv2.line(panel, (12, y), (width - 12, y), (70, 70, 70), 1)
    y += 22
    mins, secs = divmod(int(elapsed), 60)
    cv2.putText(panel, f"elapsed  {mins:02d}:{secs:02d}", (12, y), FONT, 0.42,
                (180, 180, 180), 1)
    y += 20
    cv2.putText(panel, f"fps      {fps:.1f}", (12, y), FONT, 0.42, (180, 180, 180), 1)
    if calibrating_sec > 0:
        y += 20
        cv2.putText(panel, f"calibrating room {calibrating_sec:.0f}s", (12, y),
                    FONT, 0.42, (0, 200, 255), 1)

    cv2.putText(panel, "press q to quit", (12, height - 12), FONT, 0.4,
                (120, 120, 120), 1)
    return panel


def main():
    seats = config.SEAT_COUNT
    labels = config.SEAT_LABELS[:seats]

    cam = VideoStream(config.CAMERA_SOURCE, config.CAMERA_NAME,
                      backend=getattr(config, "CAMERA_BACKEND", "MSMF"),
                      width=config.CAPTURE_WIDTH,
                      height=config.CAPTURE_HEIGHT).start()

    zone_crop = getattr(config, "ZONE_CROP_FACES", False)
    face_analyzer = FaceAnalyzer(max_faces=config.MAX_FACES,
                                 zones=seats if zone_crop else 0,
                                 crop_bottom=getattr(config, "ZONE_CROP_BOTTOM", 1.0))
    hand_analyzer = HandAnalyzer(max_hands=config.MAX_HANDS)
    detector = ObjectDetector(config.YOLO_MODEL, config.YOLO_CONF_THRESHOLD,
                              config.CATEGORY_CONF_MIN, config.YOLO_IMGSZ)
    background = StaticBackgroundFilter(config.BACKGROUND_CALIBRATION_SEC,
                                        config.BACKGROUND_IOU_THRESHOLD,
                                        config.BACKGROUND_MIN_SEEN_RATIO)

    managers = [AlertManager(config, seat=labels[i]) for i in range(seats)]
    camera_mgr = AlertManager(config, seat="CAM")

    frame_count = 0
    dets = []
    hand_points = []
    fps = 0.0
    last_fps_t = time.time()
    fps_frames = 0

    ok, probe = cam.read()
    actual = f"{probe.shape[1]}x{probe.shape[0]}" if ok and probe is not None else "unknown"
    print(f"Exam proctor running on {config.CAMERA_NAME} (source {config.CAMERA_SOURCE}).")
    print(f"Capture {actual}, {seats} seats: {', '.join(labels)}.")
    print("Press 'q' in the video window to quit.")

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1
            h, w = frame.shape[:2]
            for m in managers:
                m.set_frame(frame)
            camera_mgr.set_frame(frame)

            # The feed is shared, so tampering is judged once for everyone.
            blocked = camera_mgr.update_tamper(frame)

            # --- faces -> seats -------------------------------------------
            faces = (face_analyzer.analyze_zones(frame, seats) if zone_crop
                     else face_analyzer.analyze(frame))
            seat_faces = {}      # seat -> the largest face in it
            seat_face_count = [0] * seats
            for f in faces:
                # A zone-cropped face already knows its seat; a whole-frame one
                # is placed by where its centre falls.
                s = f.seat if f.seat is not None else seat_of(f.center_x, w, seats)
                seat_face_count[s] += 1
                cur = seat_faces.get(s)
                if cur is None or f.face_width > cur.face_width:
                    seat_faces[s] = f

            for s, face in seat_faces.items():
                m = managers[s]
                m.update_head_pose(face.yaw, face.pitch)
                m.update_talking(face.mar)
                m.update_gaze(
                    face.gaze_offset,
                    head_forward=abs(face.yaw) < config.YAW_THRESHOLD_DEG,
                    face_width=face.face_width,
                    yaw=face.yaw,
                )

            # --- hands -> seats -------------------------------------------
            if frame_count % config.HANDS_EVERY_N_FRAMES == 0:
                hand_points = hand_analyzer.analyze(frame)
                by_seat = {}
                for hx, hy in hand_points:
                    by_seat.setdefault(seat_of(hx, w, seats), []).append((hx, hy))
                for s in range(seats):
                    face = seat_faces.get(s)
                    near = bool(face) and hand_near_face(
                        by_seat.get(s, []), face.anchors, face.face_width,
                        config.HAND_NEAR_FACE_RATIO)
                    managers[s].update_hand_near_face(near)

            # --- objects -> seats -----------------------------------------
            if frame_count % config.DETECT_EVERY_N_FRAMES == 0:
                dets = detector.detect(frame)

                calibrating = background.calibrating()
                if calibrating:
                    background.observe(dets)
                else:
                    dets = background.filter(dets)

                per_seat = {s: [] for s in range(seats)}
                for d in dets:
                    per_seat[seat_of((d.box[0] + d.box[2]) / 2.0, w, seats)].append(d)

                for s in range(seats):
                    mine = per_seat[s]
                    bodies = count_label(mine, "person")
                    if not calibrating:
                        managers[s].update_objects(
                            phone_seen=has_label(mine, "phone"),
                            notes_seen=has_label(mine, "notes"),
                            device_seen=has_label(mine, "device"),
                        )
                        managers[s].update_person_count(bodies, seat_face_count[s])
                    if not blocked:
                        managers[s].update_presence(
                            s in seat_faces or bodies > 0)

            # --- draw ------------------------------------------------------
            draw_seat_zones(frame, seats, labels)
            draw_detections(frame, dets)
            draw_faces(frame, seat_faces, labels)
            for hx, hy in hand_points:
                cv2.circle(frame, (int(hx), int(hy)), 2, (200, 255, 0), -1)

            video = cv2.resize(frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))

            banner = []
            for i, m in enumerate(managers):
                a = m.get_active_banner()
                if a:
                    banner.append(f"{labels[i]}: {a}")
            cam_banner = camera_mgr.get_active_banner()
            if cam_banner:
                banner.insert(0, cam_banner)
            draw_banner(video, " | ".join(banner))

            fps_frames += 1
            if time.time() - last_fps_t >= 1.0:
                fps = fps_frames / (time.time() - last_fps_t)
                fps_frames, last_fps_t = 0, time.time()

            panel = draw_panel(managers, camera_mgr, labels, config.DISPLAY_HEIGHT,
                               config.PANEL_WIDTH, fps,
                               time.time() - managers[0].session_start,
                               background.remaining())

            cv2.imshow(config.WINDOW_NAME, cv2.hconcat([video, panel]))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.stop()
        face_analyzer.close()
        hand_analyzer.close()
        cv2.destroyAllWindows()
        print("\n--- session summary ---")
        for m in managers:
            print(m.summary())
        print(camera_mgr.summary())
        print(f"\nBackground: {background.summary()}")
        if config.ENABLE_CSV_LOG:
            print(f"Incidents logged to {config.LOG_PATH} (seat column separates students)")
        if config.SAVE_SNAPSHOTS:
            print(f"Snapshots saved to {config.SNAPSHOT_DIR}/")


if __name__ == "__main__":
    main()
