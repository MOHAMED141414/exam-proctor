import time

import cv2
import numpy as np

import config
from alert_manager import AlertManager
from camera_stream import VideoStream
from face_analyzer import FaceAnalyzer, HandAnalyzer, hand_near_face
from object_detector import ObjectDetector, count_label, has_label

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), d.color, 2)
        cv2.putText(frame, f"{d.label} {d.conf:.2f}", (x1, max(12, y1 - 8)),
                    FONT, 0.5, d.color, 2)


def draw_face_info(frame, face_result):
    h = frame.shape[0]
    if not face_result.face_found:
        cv2.putText(frame, "No face detected", (10, h - 15), FONT, 0.5, (0, 0, 255), 1)
        return

    if face_result.face_box:
        x1, y1, x2, y2 = face_result.face_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
    if face_result.nose_2d:
        cv2.circle(frame, face_result.nose_2d, 4, (255, 255, 0), -1)

    # These raw numbers are what you tune the thresholds against, so they stay
    # on screen rather than hiding behind a debug flag.
    cv2.putText(
        frame,
        f"yaw:{face_result.yaw:.1f} pitch:{face_result.pitch:.1f} "
        f"gaze:{face_result.gaze_offset:.2f} faces:{face_result.face_count}",
        (10, h - 15), FONT, 0.5, (255, 255, 0), 1,
    )


def draw_banner(frame, text):
    w = frame.shape[1]
    if text:
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 200), -1)
        # Several alerts can be live at once, so the banner may overflow the
        # window width. Truncate rather than let putText run off the edge.
        shown = text if len(text) < 58 else text[:55] + "..."
        cv2.putText(frame, f"ALERT: {shown}", (10, 27), FONT, 0.6, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (0, 0), (w, 30), (0, 120, 0), -1)
        cv2.putText(frame, "Monitoring...", (10, 21), FONT, 0.55, (255, 255, 255), 1)


def draw_panel(status, height, width, fps, elapsed):
    """Right-hand column: every check, its live state, and how many times it
    has tripped. Shows the whole picture at a glance, which the single banner
    cannot do once more than one thing is firing."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (28, 28, 28)

    cv2.putText(panel, "CHECKS", (12, 26), FONT, 0.6, (255, 255, 255), 2)
    cv2.line(panel, (12, 36), (width - 12, 36), (70, 70, 70), 1)

    y = 62
    for label, active, count in status:
        color = (60, 60, 255) if active else ((90, 190, 90) if count == 0 else (0, 165, 255))
        cv2.circle(panel, (20, y - 5), 5, color, -1)
        text = label if len(label) <= 24 else label[:23] + "."
        cv2.putText(panel, text, (34, y), FONT, 0.42, (230, 230, 230), 1)
        cv2.putText(panel, str(count), (width - 30, y), FONT, 0.42, color, 1)
        y += 26

    y += 10
    cv2.line(panel, (12, y), (width - 12, y), (70, 70, 70), 1)
    y += 24
    mins, secs = divmod(int(elapsed), 60)
    cv2.putText(panel, f"elapsed  {mins:02d}:{secs:02d}", (12, y), FONT, 0.45, (180, 180, 180), 1)
    y += 22
    cv2.putText(panel, f"fps      {fps:.1f}", (12, y), FONT, 0.45, (180, 180, 180), 1)
    y += 22
    cv2.putText(panel, f"camera   {config.CAMERA_SOURCE}", (12, y), FONT, 0.45, (180, 180, 180), 1)

    cv2.putText(panel, "press q to quit", (12, height - 14), FONT, 0.42, (120, 120, 120), 1)
    return panel


def main():
    cam = VideoStream(config.CAMERA_SOURCE, config.CAMERA_NAME,
                      backend=getattr(config, "CAMERA_BACKEND", "MSMF")).start()

    face_analyzer = FaceAnalyzer(mar_window_sec=config.MAR_STD_WINDOW_SEC,
                                 max_faces=config.MAX_FACES)
    hand_analyzer = HandAnalyzer()
    detector = ObjectDetector(config.YOLO_MODEL, config.YOLO_CONF_THRESHOLD)
    alerts = AlertManager(config)

    frame_count = 0
    dets = []
    hand_points = []
    fps = 0.0
    last_fps_t = time.time()
    fps_frames = 0

    print(f"Exam proctor running on {config.CAMERA_NAME} (source {config.CAMERA_SOURCE}).")
    print("Press 'q' in the video window to quit.")

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1
            alerts.set_frame(frame)

            # Tamper check first: on a dead or covered feed every downstream
            # detector would report "nothing there", which is indistinguishable
            # from the student having left. Checking the feed itself tells the
            # two apart, and suppresses the absence alert while blocked.
            blocked = alerts.update_tamper(frame)

            face_result = face_analyzer.analyze(frame)
            if face_result.face_found:
                alerts.update_head_pose(face_result.yaw, face_result.pitch)
                alerts.update_talking(face_result.mar_std)
                alerts.update_gaze(
                    face_result.gaze_offset,
                    head_forward=abs(face_result.yaw) < config.YAW_THRESHOLD_DEG,
                )

            # Hands are the most expensive check per frame, so throttle them.
            if frame_count % config.HANDS_EVERY_N_FRAMES == 0:
                hand_points = hand_analyzer.analyze(frame)
                alerts.update_hand_near_face(
                    hand_near_face(hand_points, face_result.anchors,
                                   face_result.face_width, config.HAND_NEAR_FACE_RATIO)
                )

            # YOLO runs every Nth frame to stay real-time on CPU; the boxes
            # from the last run are reused for drawing in between.
            if frame_count % config.DETECT_EVERY_N_FRAMES == 0:
                dets = detector.detect(frame)
                person_count = count_label(dets, "person")

                alerts.update_objects(
                    phone_seen=has_label(dets, "phone"),
                    notes_seen=has_label(dets, "notes"),
                    device_seen=has_label(dets, "device"),
                )
                alerts.update_person_count(person_count, face_result.face_count)
                if not blocked:
                    alerts.update_presence(face_result.face_found or person_count > 0)

            # Overlays go on the full-resolution frame before the resize,
            # otherwise the box coordinates would no longer line up.
            draw_detections(frame, dets)
            draw_face_info(frame, face_result)
            for hx, hy in hand_points:
                cv2.circle(frame, (int(hx), int(hy)), 2, (200, 255, 0), -1)

            video = cv2.resize(frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
            draw_banner(video, alerts.get_active_banner())

            fps_frames += 1
            if time.time() - last_fps_t >= 1.0:
                fps = fps_frames / (time.time() - last_fps_t)
                fps_frames, last_fps_t = 0, time.time()

            panel = draw_panel(alerts.get_status(), config.DISPLAY_HEIGHT,
                               config.PANEL_WIDTH, fps,
                               time.time() - alerts.session_start)

            cv2.imshow(config.WINDOW_NAME, cv2.hconcat([video, panel]))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.stop()
        face_analyzer.close()
        hand_analyzer.close()
        cv2.destroyAllWindows()
        print("\n--- session summary ---")
        print(alerts.summary())
        if config.ENABLE_CSV_LOG:
            print(f"\nIncidents logged to {config.LOG_PATH}")
        if config.SAVE_SNAPSHOTS:
            print(f"Snapshots saved to {config.SNAPSHOT_DIR}/")


if __name__ == "__main__":
    main()
