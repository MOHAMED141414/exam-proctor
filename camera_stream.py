import threading
import time

import cv2

# Windows exposes the same cameras through more than one capture backend, and
# each backend numbers them differently: on this machine index 1 is the phone
# under MSMF but the laptop webcam under DirectShow. Pinning the backend is
# what makes CAMERA_SOURCE mean one specific camera instead of "whichever
# device OpenCV happened to enumerate first today".
BACKENDS = {
    "MSMF": cv2.CAP_MSMF,
    "DSHOW": cv2.CAP_DSHOW,
    "ANY": cv2.CAP_ANY,
}


class VideoStream:
    """Threaded reader for a camera or a network video stream.

    Runs the blocking cv2.VideoCapture.read() call on its own thread so a
    slow WiFi feed (the iPhone DroidCam stream) never stalls the main
    detection loop. Also auto-reconnects if the connection drops.
    """

    def __init__(self, source, name="camera", backend="MSMF", width=None, height=None):
        self.source = source
        self.name = name
        self.width = width
        self.height = height
        # A URL source (DroidCam over WiFi) is handled by FFMPEG, so the
        # webcam backend choice only applies to integer device indices.
        self.backend = BACKENDS.get(str(backend).upper(), cv2.CAP_ANY) \
            if isinstance(source, int) else cv2.CAP_ANY

        self.cap = self._open()
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open {name} at source: {source} "
                f"(backend={backend}). Check the camera is connected and that "
                f"no other app is holding it open."
            )

        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def _open(self):
        cap = cv2.VideoCapture(self.source, self.backend)
        # Requested, not guaranteed - a device silently keeps its own size if
        # it cannot honour this, so main reports what actually came back.
        if self.width and self.height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                # Stream dropped (common over WiFi, and normal while DroidCam
                # is sitting on its "Start DroidCam" screen) - wait, reconnect.
                time.sleep(0.5)
                self.cap.release()
                self.cap = self._open()
                continue
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1)
        self.cap.release()
