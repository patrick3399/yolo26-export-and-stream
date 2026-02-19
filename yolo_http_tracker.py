#!/usr/bin/env python3
"""
YOLO HTTP-MJPEG Tracker  v1.1.0  (Model-Driven Backend + Full-Task Edition)

Overview:
    Step 2 in the two-tool pipeline.
    Takes a YOLO model exported by yolo_env_checker.py (or any supported
    format) and streams annotated video over HTTP as an MJPEG feed.
    Supports all three YOLO task types automatically:
        Detection / Tracking  — corner-box overlay + trajectory trails
        Pose Estimation       — 17-keypoint skeleton with colour-coded limbs
        Segmentation          — per-instance colour masks + corner-box overlay

Backend selection (unchanged from v1.0.0):
    The inference backend is determined automatically from the model path:
        *.pt / *.pth / *.yaml   → PyTorch  (Ultralytics auto-device)
        *.engine                → TensorRT (CUDA required)
        *_openvino_model/       → OpenVINO (folder)
        *.mlpackage / *.mlmodel → CoreML   (macOS only)
        *.onnx                  → ONNX Runtime

Task detection (v1.1.0):
    Task is inferred from the model filename suffix, then confirmed from
    the loaded model's own .task attribute:
        filename contains '-pose'   → pose
        filename contains '-seg'    → segment
        otherwise                   → detect

Device selection:
    --device is forwarded verbatim to Ultralytics; no pre-validation is
    performed, so any device string accepted by Ultralytics works
    (cuda, cpu, mps, xpu, intel:gpu, intel:npu, etc.).
    When --device is omitted, a sensible default is chosen per format:
        TensorRT  → cuda
        OpenVINO  → intel:cpu
        others    → Ultralytics auto-select

HTTP endpoints:
    /          → HTML viewer page with live stats overlay
    /stream    → MJPEG video stream (multipart/x-mixed-replace)
    /stats     → JSON performance stats (polled every 2 s by the viewer)

Features:
    - MJPEG streaming to multiple simultaneous clients
    - AJAX stats panel (FPS, inference time, dropped frames, client count)
    - Per-object trajectory trails (--trajectory)
    - Corner-box overlay style (less cluttered than full rectangles)
    - Pose: COCO 17-keypoint skeleton with colour-coded face/arm/leg zones
    - Segmentation: semi-transparent per-instance colour masks
    - Independent JPEG encoder thread (decouples CPU from inference thread)
    - Active frame-drop counter (stale frames are discarded before inference)
    - OS-aware RTSP backend selection (FFMPEG / MSMF / DSHOW / GStreamer / V4L2)
    - Automatic stream reconnect on read failure

Dependencies:
    Python ≥ 3.8, PyTorch ≥ 2.0, ultralytics, OpenCV
    Optional: TensorRT, OpenVINO, coremltools
"""

import argparse
import json
import os
import platform
import queue
import signal
import sys
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ────────────────────────────────────────────────────────────────────────────
# Model-format detection
# ────────────────────────────────────────────────────────────────────────────

def detect_model_format(model_path: str) -> str:
    """
    Determine the inference backend from the model path.

    Returns one of:
        'pytorch'   — *.pt / *.pth / *.yaml
        'tensorrt'  — *.engine
        'openvino'  — directory (any folder or path containing _openvino_model)
        'coreml'    — *.mlpackage or *.mlmodel
        'onnx'      — *.onnx
        'unknown'   — anything else (let Ultralytics decide)
    """
    p = str(model_path).rstrip("/\\")

    # Directory path → OpenVINO
    if os.path.isdir(p):
        return "openvino"

    lower = p.lower()
    if lower.endswith(".pt") or lower.endswith(".pth") or lower.endswith(".yaml"):
        return "pytorch"
    if lower.endswith(".engine"):
        return "tensorrt"
    if lower.endswith(".onnx"):
        return "onnx"
    if lower.endswith(".mlpackage") or lower.endswith(".mlmodel"):
        return "coreml"
    # String-pattern fallback for OpenVINO folder specified before it exists on disk
    if "_openvino_model" in lower:
        return "openvino"

    return "unknown"


def format_display_name(fmt: str) -> str:
    """Return a human-readable backend name."""
    return {
        "pytorch":  "PyTorch",
        "tensorrt": "TensorRT (.engine)",
        "openvino": "OpenVINO (folder)",
        "coreml":   "CoreML (.mlpackage)",
        "onnx":     "ONNX Runtime (.onnx)",
        "unknown":  "Unknown (Ultralytics auto)",
    }.get(fmt, fmt)


def infer_precision(fmt: str) -> str:
    """Best-effort precision label used in the status overlay."""
    return {
        "tensorrt": "depends on .engine",
        "onnx":     "fp32/fp16",
        "openvino": "depends on export",
        "coreml":   "depends on export",
        "pytorch":  "fp32 (auto)",
        "unknown":  "auto",
    }.get(fmt, "auto")


def detect_model_task(model_path: str) -> str:
    """
    Infer the YOLO task from the model filename.

    Looks for well-known suffixes in the stem (before the extension):
        contains '-pose'  → 'pose'
        contains '-seg'   → 'segment'
        otherwise         → 'detect'

    After the model is loaded, the caller should also check
    model.task and override this value if they differ.
    """
    stem = str(model_path).lower()
    if '-pose' in stem:
        return 'pose'
    if '-seg' in stem:
        return 'segment'
    return 'detect'


def task_display_name(task: str) -> str:
    """Return a short human-readable task label for the info overlay."""
    return {
        'detect':  'Detection / Tracking',
        'pose':    'Pose Estimation',
        'segment': 'Segmentation',
    }.get(task, task)


# ── COCO 17-keypoint skeleton definition ─────────────────────────────────────
# Each tuple is (keypoint_index_A, keypoint_index_B, BGR_colour).
# Keypoint order: 0=nose 1=left_eye 2=right_eye 3=left_ear 4=right_ear
#                 5=left_shoulder 6=right_shoulder 7=left_elbow 8=right_elbow
#                 9=left_wrist 10=right_wrist 11=left_hip 12=right_hip
#                13=left_knee 14=right_knee 15=left_ankle 16=right_ankle
POSE_SKELETON = [
    # Face
    (0, 1,  (255, 220,  50)),
    (0, 2,  (255, 220,  50)),
    (1, 3,  (255, 220,  50)),
    (2, 4,  (255, 220,  50)),
    # Shoulder bar
    (5, 6,  (0, 200, 255)),
    # Left arm
    (5, 7,  (0, 200, 255)),
    (7, 9,  (0, 200, 255)),
    # Right arm
    (6, 8,  (255, 100, 100)),
    (8, 10, (255, 100, 100)),
    # Torso
    (5, 11, (180, 180, 180)),
    (6, 12, (180, 180, 180)),
    # Hip bar
    (11, 12, (180, 180, 180)),
    # Left leg
    (11, 13, (100, 255, 100)),
    (13, 15, (100, 255, 100)),
    # Right leg
    (12, 14, (50, 180, 255)),
    (14, 16, (50, 180, 255)),
]

# Keypoint confidence threshold — points below this are not drawn
POSE_KP_CONF = 0.3

# Pre-group POSE_SKELETON connections by BGR colour so the draw loop can issue
# one cv2.polylines() call per colour instead of one cv2.line() per connection.
# Built once at module import time; never mutated at runtime.
from collections import defaultdict as _defaultdict
_SKELETON_BY_COLOR: dict = _defaultdict(list)
for _a, _b, _c in POSE_SKELETON:
    _SKELETON_BY_COLOR[_c].append((_a, _b))
del _a, _b, _c  # clean up loop variables from module namespace

# Maximum wall-clock seconds allowed for a single cap.read() call.
# When the RTSP/ffmpeg backend hangs (no return, no error), this timeout
# forces _update() to treat the situation as a read failure and trigger
# the reconnect logic instead of blocking the thread indefinitely.
_CAP_READ_TIMEOUT = 5.0

# ── Segmentation mask colour palette (BGR, 20 distinct colours cycling by class)
SEG_PALETTE = [
    (  0, 200, 255), (  0, 255, 100), (255, 100,  50), (200,   0, 255),
    (255, 220,   0), ( 50, 255, 200), (255,  50, 200), (  0, 150, 255),
    (100, 255,  50), (255, 150,   0), (  0, 255, 200), (150,  50, 255),
    (255,   0, 100), ( 50, 200, 100), (200, 255,   0), (  0, 100, 200),
    (255, 200, 100), (100,   0, 255), (  0, 255,  50), (200, 100, 255),
]


# ────────────────────────────────────────────────────────────────────────────
# Performance monitor
# ────────────────────────────────────────────────────────────────────────────

class PerformanceMonitor:
    """Records per-category timings and frame / drop statistics."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.times = {
            k: deque(maxlen=window_size)
            for k in ("read_frame", "inference", "draw", "encode", "total")
        }
        self.lock = threading.Lock()
        self.frame_count = 0
        self.dropped_frames = 0
        self.current_fps = 0.0
        self.fps_lock = threading.Lock()
        self.client_count = 0
        self.client_lock = threading.Lock()

    def record(self, category: str, duration_ms: float):
        with self.lock:
            self.times[category].append(duration_ms)

    def record_frame(self):
        with self.lock:
            self.frame_count += 1

    def record_drop(self):
        with self.lock:
            self.dropped_frames += 1

    def update_fps(self, fps: float):
        with self.fps_lock:
            self.current_fps = fps

    def get_fps(self) -> float:
        with self.fps_lock:
            return self.current_fps

    def add_client(self):
        with self.client_lock:
            self.client_count += 1

    def remove_client(self):
        with self.client_lock:
            self.client_count = max(0, self.client_count - 1)

    def get_client_count(self) -> int:
        with self.client_lock:
            return self.client_count

    def get_stats(self) -> dict:
        with self.lock:
            stats = {}
            for cat, series in self.times.items():
                if series:
                    arr = np.array(series, dtype=float)
                    stats[cat] = {
                        "avg": float(arr.mean()),
                        "min": float(arr.min()),
                        "max": float(arr.max()),
                        "std": float(arr.std()),
                        "current": float(series[-1]),
                    }
                else:
                    stats[cat] = {"avg": 0, "min": 0, "max": 0, "std": 0, "current": 0}

            stats["frame_count"]    = self.frame_count
            stats["dropped_frames"] = self.dropped_frames
            stats["drop_rate"]      = (
                self.dropped_frames / self.frame_count * 100.0
                if self.frame_count > 0 else 0.0
            )
            stats["client_count"] = self.get_client_count()
            stats["fps"]          = self.get_fps()
        return stats

    def print_report(self):
        s = self.get_stats()
        print("\n" + "=" * 70)
        print("📊 PERFORMANCE REPORT (v1.1.0 Full-Task Edition)")
        print("=" * 70)
        print(f"Current FPS : {s['fps']:.1f}")
        print(f"Clients     : {s['client_count']}")
        for cat in ("read_frame", "inference", "draw", "encode", "total"):
            d = s[cat]
            print(
                f"  {cat:13s}: avg={d['avg']:6.1f}ms  "
                f"min={d['min']:6.1f}ms  max={d['max']:6.1f}ms  std={d['std']:5.1f}ms"
            )
        print(f"Dropped     : {s['dropped_frames']} ({s['drop_rate']:.1f}%)")
        print("=" * 70)


# ────────────────────────────────────────────────────────────────────────────
# RTSP stream loader
# ────────────────────────────────────────────────────────────────────────────

class RTSPStreamLoader:
    """
    OS-aware RTSP / webcam reader with active frame-drop handling.

    Tries multiple OpenCV backends in order of preference for the current OS:
        Windows : CAP_FFMPEG → CAP_MSMF → CAP_DSHOW → CAP_ANY
        Linux   : CAP_FFMPEG → CAP_GSTREAMER → CAP_V4L2
    Automatically reconnects if the stream drops.
    Stale frames are discarded from the internal queue so inference always
    receives the most recent available frame.
    """

    def __init__(self, src, monitor: PerformanceMonitor):
        self.src = src
        self.monitor = monitor
        self.stop_event = threading.Event()
        self.q: queue.Queue = queue.Queue(maxsize=2)
        self.reconnect_delay = 5

        system_os = platform.system()
        if system_os == "Windows":
            self.backend_list = [
                ("CAP_FFMPEG", cv2.CAP_FFMPEG),
                ("CAP_MSMF",   cv2.CAP_MSMF),
                ("CAP_DSHOW",  cv2.CAP_DSHOW),
                ("CAP_ANY",    cv2.CAP_ANY),
            ]
            print(f"🪟 Windows — backends: {[b[0] for b in self.backend_list]}")
        elif system_os == "Darwin":
            self.backend_list = [
                ("CAP_AVFOUNDATION", cv2.CAP_AVFOUNDATION),
                ("CAP_FFMPEG",       cv2.CAP_FFMPEG),
                ("CAP_ANY",          cv2.CAP_ANY),
            ]
            print(f"🍎 macOS — backends: {[b[0] for b in self.backend_list]}")
        else:
            self.backend_list = [
                ("CAP_FFMPEG",    cv2.CAP_FFMPEG),
                ("CAP_GSTREAMER", cv2.CAP_GSTREAMER),
                ("CAP_V4L2",      cv2.CAP_V4L2),
            ]
            print(f"🐧 Linux — backends: {[b[0] for b in self.backend_list]}")

        self.cap = None
        self.connected = False
        self.backend_name = None
        # L3: slot for the currently active cap.read() daemon thread.
        # At most ONE thread is allowed per cap instance (see _timed_read).
        self._pending_read_thread: Optional[threading.Thread] = None

        self._open_capture()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _open_capture(self):
        """Try each backend in order; raise RuntimeError if all fail.

        L2 fix: release the old cap *before* opening a new one.
        --------------------------------------------------------
        Any daemon thread currently blocked inside C++ ``cap.read()`` holds a
        reference to the previous VideoCapture object.  Calling
        ``cap.release()`` on the old object closes the underlying socket /
        file-descriptor, which causes the C++ read to return an error and
        allows the zombie thread to exit naturally — instead of waiting for
        an OS-level TCP timeout that can take tens of minutes.
        """
        # Release the previous capture so zombie threads can unblock
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        for name, code in self.backend_list:
            try:
                print(f"   Trying {name} ...", end="", flush=True)
                cap = cv2.VideoCapture(self.src, code)
                if code == cv2.CAP_FFMPEG:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Minimise latency
                    # L1 fix: ask FFmpeg to enforce its own read timeout at the
                    # C level.  When these properties are honoured (OpenCV ≥ 4.1
                    # built with a sufficiently recent FFmpeg), cap.read() will
                    # raise an error rather than block forever, so _timed_read()
                    # never needs to rely on the daemon-thread sentinel at all.
                    timeout_ms = int(_CAP_READ_TIMEOUT * 1000)
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.cap = cap
                        self.backend_name = name
                        self.connected = True
                        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        print(f" ✔ ({w}×{h} @ {fps:.1f} FPS)")
                        return
                    print(" ✗ (read failed)")
                else:
                    print(" ✗ (open failed)")
                cap.release()
            except Exception as e:
                print(f" ✗ ({e})")
        raise RuntimeError(f"Unable to open stream: {self.src}")

    def _timed_read(self) -> tuple:
        """
        Call ``self.cap.read()`` in a daemon thread with a wall-clock timeout.

        Three-layer defence against thread leaks
        ----------------------------------------
        L1 (OpenCV-native, set in _open_capture):
            ``CAP_PROP_READ_TIMEOUT_MSEC`` asks the FFmpeg backend to honour a
            read deadline inside C++.  When supported this prevents any hang
            entirely, making this thread-based fallback unnecessary.

        L2 (cap.release in _open_capture):
            Before creating a replacement VideoCapture, _open_capture() calls
            ``self.cap.release()`` on the old object.  That closes the
            underlying socket, which causes a hung C++ ``cap.read()`` to
            return an error, letting the zombie thread exit naturally.

        L3 (one-thread-per-cap, implemented here):
            ``self._pending_read_thread`` records the current daemon thread.
            If it is still alive when we are called again, the cap is still
            stuck: we return failure immediately *without spawning a second
            thread*.  This bounds the worst-case leak to exactly one thread
            per VideoCapture instance — i.e. O(reconnects), not O(frames).

        Race-condition note
        -------------------
        ``self.cap`` is captured into a local ``cap_snapshot`` before the
        thread is started.  Even if ``_open_capture()`` reassigns ``self.cap``
        to a new object on another code path, the running thread continues to
        operate on the original cap and never touches the new one.
        """
        # L3: if the previous thread is still alive the cap is stuck — avoid
        # spawning another thread on top of the already-stuck one.
        if (self._pending_read_thread is not None
                and self._pending_read_thread.is_alive()):
            return False, None

        # Snapshot the cap reference so the closure is immune to reassignment
        # of self.cap that may happen concurrently during a reconnect cycle.
        cap_snapshot = self.cap
        result = [False, None]
        done   = threading.Event()

        def _do_read():
            result[0], result[1] = cap_snapshot.read()
            done.set()

        t = threading.Thread(target=_do_read, daemon=True, name="rtsp-read")
        self._pending_read_thread = t
        t.start()
        if done.wait(_CAP_READ_TIMEOUT):
            # Read completed within the deadline — clear the slot.
            self._pending_read_thread = None
            return result[0], result[1]
        # Timed out.  _update() will set self.connected=False; _open_capture()
        # will call cap_snapshot.release(), which should unblock _do_read().
        print(f"⚠ cap.read() timed out after {_CAP_READ_TIMEOUT:.0f}s — reconnecting …")
        return False, None

    def _update(self):
        """Background thread: continuously read frames and keep queue fresh."""
        while not self.stop_event.is_set():
            if not self.connected:
                time.sleep(self.reconnect_delay)
                try:
                    self._open_capture()
                    print("✔ Stream reconnected!")
                except Exception:
                    continue
                continue

            t0 = time.time()
            ret, frame = self._timed_read()
            self.monitor.record("read_frame", (time.time() - t0) * 1000.0)

            if not ret:
                print("⚠ Read failed — reconnecting …")
                self.connected = False
                continue

            # Drain stale frames so inference sees the latest one
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                    self.monitor.record_drop()
                except queue.Empty:
                    break

            try:
                self.q.put(frame, block=False)
            except queue.Full:
                try:
                    self.q.get_nowait()
                    self.q.put(frame, block=False)
                except Exception:
                    self.monitor.record_drop()

    def read(self):
        """Return the most recent frame, or None if none arrives within 1 s."""
        try:
            return self.q.get(timeout=1.0)
        except queue.Empty:
            return None

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()


# ────────────────────────────────────────────────────────────────────────────
# HTTP server
# ────────────────────────────────────────────────────────────────────────────

class StreamingHandler(BaseHTTPRequestHandler):
    """Serves the MJPEG stream, the HTML viewer page, and JSON stats."""

    def do_GET(self):
        if self.path == "/":
            self._serve_html()
        elif self.path == "/stream":
            self._serve_stream()
        elif self.path == "/stats":
            self._serve_stats()
        else:
            self.send_error(404)

    # ------------------------------------------------------------------
    def _serve_html(self):
        """Serve the self-contained HTML viewer with an auto-refresh stats panel."""
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()

        tracker = self.server.tracker
        stats = tracker.monitor.get_stats()

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>YOLO Tracker v1.1.0</title>
  <style>
    body {{ margin:0; background:#000; color:#fff; font-family:Arial,sans-serif; }}
    .info {{
      position:absolute; top:10px; left:10px;
      background:rgba(0,0,0,.65); padding:10px 14px;
      border-radius:6px; font-size:14px; line-height:1.7;
    }}
    .fps {{ color:#00ff88; font-weight:bold; }}
    .fmt {{ color:#7dd3fc; }}
    .task {{ color:#fbbf24; }}
  </style>
  <script>
    function refresh() {{
      fetch('/stats').then(r=>r.json()).then(d=>{{
        document.getElementById('fps-v').textContent  = d.fps.toFixed(1);
        document.getElementById('inf-v').textContent  = d.inference_avg.toFixed(1)+' ms';
        document.getElementById('drop-v').textContent = d.dropped_frames+' ('+d.drop_rate.toFixed(1)+'%)';
        document.getElementById('cli-v').textContent  = d.client_count;
      }}).catch(()=>{{}});
    }}
    setInterval(refresh, 2000);
    window.onload = refresh;
  </script>
</head>
<body>
  <img src="/stream" style="display:block;margin:0 auto;max-width:100%;height:auto;">
  <div class="info">
    <div><b>Model:</b> <span class="fmt">{tracker.model_name}</span></div>
    <div><b>Task:</b> <span class="task">{tracker.task_label}</span></div>
    <div><b>Backend:</b> <span class="fmt">{tracker.format_display}</span></div>
    <div><b>Precision:</b> {tracker.precision_label}</div>
    <div><b>Device:</b> {tracker.device_label}</div>
    <div><b>FPS:</b> <span id="fps-v" class="fps">{stats['fps']:.1f}</span></div>
    <div><b>Inference:</b> <span id="inf-v">{stats['inference']['avg']:.1f} ms</span></div>
    <div><b>Dropped:</b> <span id="drop-v">{stats['dropped_frames']} ({stats['drop_rate']:.1f}%)</span></div>
    <div><b>Clients:</b> <span id="cli-v">{stats['client_count']}</span></div>
  </div>
</body>
</html>"""
        self.wfile.write(html.encode("utf-8"))

    # ------------------------------------------------------------------
    def _serve_stats(self):
        """Serve JSON performance stats (polled by the viewer page)."""
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        s = self.server.tracker.monitor.get_stats()
        payload = {
            "fps":            s["fps"],
            "inference_avg":  s["inference"]["avg"],
            "dropped_frames": s["dropped_frames"],
            "drop_rate":      s["drop_rate"],
            "client_count":   s["client_count"],
        }
        self.wfile.write(json.dumps(payload).encode())

    # ------------------------------------------------------------------
    def _serve_stream(self):
        """Serve the MJPEG stream (multipart/x-mixed-replace boundary)."""
        self.send_response(200)
        self.send_header("Content-type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()

        self.server.tracker.monitor.add_client()
        try:
            while self.server.tracker.running:
                jpeg = self.server.tracker.get_jpeg()
                if jpeg is None:
                    time.sleep(0.01)
                    continue
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-type", "image/jpeg")
                    self.send_header("Content-length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    break
        finally:
            self.server.tracker.monitor.remove_client()

    def log_message(self, fmt, *args):
        pass  # Suppress per-request log noise


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


# ────────────────────────────────────────────────────────────────────────────
# Main tracker
# ────────────────────────────────────────────────────────────────────────────

class YOLOTracker:
    """
    YOLO HTTP-MJPEG Tracker  v1.1.0

    Combines model-format detection, task detection, RTSP/webcam reading,
    YOLO inference, annotation, and HTTP serving into a single pipeline.

    Task detection:
        Inferred from model filename suffix; confirmed from model.task after load.
        'detect'  → corner-box + labels + optional trajectories
        'pose'    → COCO 17-keypoint skeleton overlay
        'segment' → semi-transparent colour masks + corner-box + labels

    Device resolution:
        1. Detect backend from model path (detect_model_format()).
        2. If --device provided → pass verbatim to every predict/track call.
           No validation; Ultralytics raises a clear error for invalid values.
        3. If --device NOT provided (empty string):
           • pytorch / unknown  → omit device kwarg (Ultralytics auto-select)
           • tensorrt           → device="cuda"
           • openvino           → device="intel:cpu"
           • coreml / onnx      → omit device kwarg
    """

    def __init__(self, args):
        self.args = args
        self.running = False
        self.model_name  = args.model
        self.tracker_cfg = args.tracker
        self.port        = args.port
        self.conf_thres  = args.conf
        self.iou_thres   = args.iou
        self.frame_skip  = max(1, args.frame_skip)
        self.quality     = args.quality
        self.show_id     = args.show_id
        self.trajectory  = args.trajectory
        self.trajectory_length = args.trajectory_length
        self.filter_classes    = args.classes or []
        self.input_src         = args.input

        # ── Detect model format ──────────────────────────────────────────────
        self.model_format    = detect_model_format(self.model_name)
        self.format_display  = format_display_name(self.model_format)
        self.precision_label = infer_precision(self.model_format)

        # ── Detect model task (confirmed after load) ─────────────────────────
        self.model_task  = detect_model_task(self.model_name)
        self.task_label  = task_display_name(self.model_task)

        # ── Resolve inference device kwarg ───────────────────────────────────
        user_device = args.device.strip()
        if user_device:
            # User explicitly set --device; forward verbatim
            self._device_kwarg = user_device
            self.device_label  = user_device
        else:
            # Auto-resolve per format
            defaults = {
                "tensorrt": "cuda",
                "openvino": "intel:cpu",
            }
            auto = defaults.get(self.model_format, None)
            self._device_kwarg = auto          # None → omit from kwargs
            self.device_label  = auto or "auto (Ultralytics)"

        # ── Pipeline state ───────────────────────────────────────────────────
        self.monitor       = PerformanceMonitor()
        self.frame_lock    = threading.Lock()
        self.latest_frame  = None
        self.latest_jpeg   = None
        self.stop_encode_event = threading.Event()
        self.encode_thread = None

        self.model       = None
        self.class_names = {}
        self.filter_indices: list = []

        self._print_diagnostic()
        self._load_model()

        if self.trajectory:
            self.trajectories: dict = defaultdict(
                lambda: deque(maxlen=self.trajectory_length)
            )

        self._start_encoding_thread()

    # ──────────────────────────────────────────────────────────────────────────
    def _print_diagnostic(self):
        print("\n" + "=" * 70)
        print("🔍 SYSTEM DIAGNOSTIC (v1.1.0 Model-Driven Backend + Full-Task)")
        print("=" * 70)
        print(f"Python     : {sys.version.split()[0]}")
        print(f"PyTorch    : {torch.__version__}")
        print(f"CUDA avail : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Platform   : {platform.system()} {platform.machine()}")
        print("-" * 70)
        print(f"Model      : {self.model_name}")
        print(f"Task       : {self.task_label}")
        print(f"Format     : {self.format_display}")
        print(f"Device     : {self.device_label}")
        print(f"Precision  : {self.precision_label}")
        print("=" * 70 + "\n")

    # ──────────────────────────────────────────────────────────────────────────
    def _build_device_kwargs(self) -> dict:
        """Return {'device': value} if a device is set, else empty dict."""
        if self._device_kwarg is not None:
            return {"device": self._device_kwarg}
        return {}

    # ──────────────────────────────────────────────────────────────────────────
    def _load_model(self):
        """
        Load the YOLO model and run a lightweight warm-up pass.
        After loading, the model's own .task attribute is used to confirm
        (or correct) the task inferred from the filename.
        """
        print(f"📦 Loading model: {self.model_name}  [{self.format_display}]")

        self.model = YOLO(self.model_name)

        # Confirm task from the loaded model (more reliable than filename)
        model_reported_task = getattr(self.model, 'task', None)
        if model_reported_task and model_reported_task != self.model_task:
            print(f"   ℹ Task corrected by model metadata: "
                  f"'{self.model_task}' → '{model_reported_task}'")
            self.model_task = model_reported_task
            self.task_label = task_display_name(self.model_task)

        print(f"   Task: {self.task_label}")

        # Build class-name index for filtering
        self.class_names = self.model.names
        if self.filter_classes:
            name2idx = {v: k for k, v in self.class_names.items()}
            for cls_name in self.filter_classes:
                if cls_name in name2idx:
                    self.filter_indices.append(name2idx[cls_name])
                else:
                    print(f"⚠ Unknown class name: '{cls_name}'")

        # Warm-up pass — let Ultralytics handle all device/precision details
        print("🔥 Running warm-up pass …")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        device_kw = self._build_device_kwargs()
        t0 = time.time()
        try:
            self.model.predict(dummy, verbose=False, **device_kw)
            ms = (time.time() - t0) * 1000.0
            self.monitor.record("inference", ms)
            print(f"   ✔ Warm-up done in {ms:.1f} ms")
        except Exception as e:
            print(f"   ⚠ Warm-up failed: {e}")
            print("     Continuing anyway — inference may still succeed on first frame.")

    # ──────────────────────────────────────────────────────────────────────────
    def _start_encoding_thread(self):
        """Start a background thread that converts frames to JPEG bytes."""
        def _loop():
            while not self.stop_encode_event.is_set():
                with self.frame_lock:
                    if self.latest_frame is None:
                        frame = None
                    else:
                        frame = self.latest_frame.copy()
                if frame is None:
                    time.sleep(0.01)
                    continue
                t0 = time.time()
                ok, buf = cv2.imencode(
                    ".jpg", frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.quality],
                )
                self.monitor.record("encode", (time.time() - t0) * 1000.0)
                if ok:
                    with self.frame_lock:
                        self.latest_jpeg = buf.tobytes()

        self.encode_thread = threading.Thread(target=_loop, daemon=True)
        self.encode_thread.start()
        print("✔ Encoding thread started")

    # ──────────────────────────────────────────────────────────────────────────
    def get_jpeg(self):
        """Return the latest encoded JPEG bytes (thread-safe)."""
        with self.frame_lock:
            return self.latest_jpeg

    # ──────────────────────────────────────────────────────────────────────────
    def _draw_corner_box(self, img, x1, y1, x2, y2,
                         color=(0, 255, 0), thickness=4):
        """
        Draw a corner-style bounding box (less cluttered than a full rectangle).
        Corner length is 18 % of the shorter box dimension.

        Optimisation: all 8 line segments are encoded as four L-shaped
        polylines and drawn with a single cv2.polylines() call, eliminating
        7 redundant Python→C boundary crossings per bounding box compared to
        the original 8 × cv2.line() approach.
        """
        ll = max(15, int(min(x2 - x1, y2 - y1) * 0.18))
        corners = np.array([
            [[x1 + ll, y1], [x1,      y1], [x1,      y1 + ll]],  # top-left
            [[x2 - ll, y1], [x2,      y1], [x2,      y1 + ll]],  # top-right
            [[x2 - ll, y2], [x2,      y2], [x2,      y2 - ll]],  # bottom-right
            [[x1 + ll, y2], [x1,      y2], [x1,      y2 - ll]],  # bottom-left
        ], dtype=np.int32)
        cv2.polylines(img, corners, False, color, thickness)

    def _draw_trajectory(self, img, track_id, center,
                         color=(0, 255, 0)):
        """
        Draw a trajectory trail for a tracked object.

        Optimisation: instead of calling cv2.line() once per segment in a
        Python loop (up to trajectory_length-1 calls), all trail points are
        packed into a single (N, 1, 2) int32 numpy array and drawn with one
        cv2.polylines() call.

        Fading note: the original loop varied line thickness with alpha
        (max(1, int(2*alpha))), but for the default trajectory_length=30
        the computed thickness is always 1 (alpha never reaches 1.0 inside
        the loop).  The simplified single-polylines approach preserves the
        same visual result with a fraction of the Python overhead.
        """
        self.trajectories[track_id].append(center)
        pts = list(self.trajectories[track_id])
        if len(pts) >= 2:
            pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts_arr], False, color, 2)
        cv2.circle(img, (int(center[0]), int(center[1])), 4, color, -1)

    def _draw_detections(self, img, results) -> int:
        """Annotate the frame with corner-boxes, labels, and optional trajectories."""
        if not results or not results[0].boxes:
            return 0

        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.8, 2

        # Pre-compute all detection data and draw non-overlay elements
        det_data = []
        for b in results[0].boxes:
            cls_id = int(b.cls[0])
            conf   = float(b.conf[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())

            label = (
                f"#{int(b.id[0])} {self.class_names[cls_id]} {int(conf*100)}%"
                if self.show_id and b.id is not None
                else f"{self.class_names[cls_id]} {int(conf*100)}%"
            )

            if self.trajectory and b.id is not None:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                self._draw_trajectory(img, int(b.id[0]), (cx, cy))

            self._draw_corner_box(img, x1, y1, x2, y2)

            (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
            det_data.append((x1, y1, tw, th, label))

        # Single overlay for all semi-transparent label backgrounds
        overlay = img.copy()
        for x1, y1, tw, th, _ in det_data:
            cv2.rectangle(overlay, (x1, y1 - th - 12), (x1 + tw + 12, y1), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        # Draw text on top of the blended image
        for x1, y1, _, _, label in det_data:
            cv2.putText(img, label, (x1 + 6, y1 - 6), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

        return len(det_data)

    # ──────────────────────────────────────────────────────────────────────────
    def _draw_pose(self, img, results) -> int:
        """
        Draw COCO 17-keypoint pose skeletons on the frame.

        For each detected person:
          1. Draw skeleton limb connections (colour-coded by body zone).
          2. Draw a filled circle at each visible keypoint.
          3. Draw the bounding box and label (same style as detection).

        Keypoints below POSE_KP_CONF confidence are skipped entirely so that
        partially-visible people do not produce phantom limbs.
        """
        if not results or not results[0].keypoints:
            return 0

        r      = results[0]
        kps    = r.keypoints          # shape: (N, 17, 3)  x, y, conf
        boxes  = r.boxes if r.boxes is not None else []
        num    = 0

        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.7, 2
        label_data = []  # collect label rects for single-overlay blending

        for person_idx in range(len(kps)):
            kp_data = kps[person_idx].data[0].cpu().numpy()  # (17, 3)

            # ── Draw skeleton limbs (optimised) ────────────────────────────
            # Group valid segments by colour and issue one cv2.polylines()
            # call per colour instead of one cv2.line() per connection.
            # POSE_SKELETON has 6 distinct colours → at most 6 cv2 calls
            # per person, down from 16.
            for color, connections in _SKELETON_BY_COLOR.items():
                segs = []
                for a, b in connections:
                    if kp_data[a][2] < POSE_KP_CONF or kp_data[b][2] < POSE_KP_CONF:
                        continue
                    segs.append(np.array([
                        [[int(kp_data[a][0]), int(kp_data[a][1])]],
                        [[int(kp_data[b][0]), int(kp_data[b][1])]],
                    ], dtype=np.int32))
                if segs:
                    cv2.polylines(img, segs, False, color, 2, cv2.LINE_AA)

            # ── Draw keypoint circles ───────────────────────────────────────
            # Use numpy masking to filter valid keypoints before entering the
            # Python loop, avoiding per-keypoint `if` evaluation overhead.
            valid_mask = kp_data[:, 2] >= POSE_KP_CONF
            for kp_idx in np.where(valid_mask)[0]:
                x, y = int(kp_data[kp_idx, 0]), int(kp_data[kp_idx, 1])
                radius = 3 if kp_idx < 5 else 5
                cv2.circle(img, (x, y), radius, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (x, y), radius, (0, 0, 0),        1, cv2.LINE_AA)

            # ── Collect bounding box and label data ─────────────────────────
            if person_idx < len(boxes):
                bx = boxes[person_idx]
                conf_val = float(bx.conf[0])
                x1, y1, x2, y2 = map(int, bx.xyxy[0].cpu().numpy())
                label = (
                    f"#{int(bx.id[0])} person {int(conf_val*100)}%"
                    if self.show_id and bx.id is not None
                    else f"person {int(conf_val*100)}%"
                )
                if self.trajectory and bx.id is not None:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    self._draw_trajectory(img, int(bx.id[0]), (cx, cy),
                                          color=(255, 220, 50))

                self._draw_corner_box(img, x1, y1, x2, y2,
                                      color=(255, 220, 50), thickness=2)
                (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
                label_data.append((x1, y1, tw, th, label))
            num += 1

        # Single overlay for all semi-transparent label backgrounds
        if label_data:
            overlay = img.copy()
            for x1, y1, tw, th, _ in label_data:
                cv2.rectangle(overlay, (x1, y1 - th - 10), (x1 + tw + 10, y1),
                              (255, 220, 50), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            for x1, y1, _, _, label in label_data:
                cv2.putText(img, label, (x1 + 5, y1 - 5),
                            font, scale, (0, 0, 0), thick, cv2.LINE_AA)

        return num

    # ──────────────────────────────────────────────────────────────────────────
    def _draw_masks(self, img, results) -> int:
        """
        Draw semi-transparent instance segmentation masks on the frame.

        For each detected instance:
          1. Retrieve the binary mask (already rescaled to input resolution
             by Ultralytics) and resize it to the original frame dimensions.
          2. Flood-fill the mask region with a per-class colour at 40 % opacity.
          3. Draw the mask contour outline for a sharper edge.
          4. Draw the corner-box and label on top (same as detection).

        Mask colours cycle through SEG_PALETTE by class_id so that objects
        of the same class always share a colour across frames.
        """
        if not results or not results[0].masks:
            return 0

        r      = results[0]
        masks  = r.masks       # Ultralytics Masks object
        boxes  = r.boxes
        h, w   = img.shape[:2]
        num    = 0

        # Build a composite overlay; blend once at the end for efficiency
        overlay = img.copy()
        contour_list = []  # collect contours to draw after blending

        for i, mask_obj in enumerate(masks):
            # masks.data is (N, mH, mW) as a float tensor in [0, 1]
            mask_np = mask_obj.data[0].cpu().numpy()
            # Resize mask to frame resolution
            mask_bin = cv2.resize(mask_np, (w, h),
                                  interpolation=cv2.INTER_LINEAR) > 0.5

            cls_id = int(boxes[i].cls[0]) if boxes is not None else i
            color  = SEG_PALETTE[cls_id % len(SEG_PALETTE)]

            overlay[mask_bin] = color

            # Collect contour outlines for drawing after the blend
            contours, _ = cv2.findContours(
                mask_bin.astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contour_list.append((contours, color))

        # Blend the filled overlay with the original frame at 40 % opacity
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

        # Draw contour outlines at full opacity (after blend for crisp edges)
        for contours, color in contour_list:
            cv2.drawContours(img, contours, -1, color, 2)

        # Draw boxes and labels on top of the blended masks
        if boxes is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale, thick = 0.75, 2
            label_data = []

            for b in boxes:
                cls_id  = int(b.cls[0])
                conf    = float(b.conf[0])
                x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
                color   = SEG_PALETTE[cls_id % len(SEG_PALETTE)]
                label   = (
                    f"#{int(b.id[0])} {self.class_names[cls_id]} {int(conf*100)}%"
                    if self.show_id and b.id is not None
                    else f"{self.class_names[cls_id]} {int(conf*100)}%"
                )
                if self.trajectory and b.id is not None:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    self._draw_trajectory(img, int(b.id[0]), (cx, cy),
                                          color=color)

                self._draw_corner_box(img, x1, y1, x2, y2,
                                      color=color, thickness=2)
                (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
                label_data.append((x1, y1, tw, th, label, color))

            # Single overlay for all semi-transparent label backgrounds
            if label_data:
                lbl_overlay = img.copy()
                for x1, y1, tw, th, _, color in label_data:
                    cv2.rectangle(lbl_overlay, (x1, y1 - th - 10),
                                  (x1 + tw + 10, y1), color, -1)
                cv2.addWeighted(lbl_overlay, 0.65, img, 0.35, 0, img)
                for x1, y1, _, _, label, _ in label_data:
                    cv2.putText(img, label, (x1 + 5, y1 - 5),
                                font, scale, (255, 255, 255), thick, cv2.LINE_AA)

            num = len(label_data)
        return num

    # ──────────────────────────────────────────────────────────────────────────
    def process_loop(self):
        """
        Main inference loop: read → infer → draw (task-specific) → update JPEG.

        Drawing is dispatched by self.model_task:
            'detect'  → _draw_detections()
            'pose'    → _draw_pose()
            'segment' → _draw_masks()  (includes box/label overlay)
        """
        print("\n" + "=" * 70)
        print(f"✔ YOLO processing started  [{self.task_label}]  (v1.1.0)")
        print("=" * 70)
        print("Press Ctrl+C to stop and view performance report\n")

        self.loader = RTSPStreamLoader(self.input_src, monitor=self.monitor)

        device_kw     = self._build_device_kwargs()
        frame_cnt     = 0
        fps_tick      = time.time()
        fps_cnt       = 0

        while self.running:
            # Pause processing when no client is connected (saves CPU/GPU)
            if self.monitor.get_client_count() == 0:
                time.sleep(0.5)
                continue

            loop_t0 = time.time()

            frame = self.loader.read()
            if frame is None:
                continue

            self.monitor.record_frame()
            frame_cnt += 1

            # Skip frames if frame_skip > 1
            if self.frame_skip > 1 and (frame_cnt % self.frame_skip) != 0:
                continue

            # ── Inference ──────────────────────────────────────────────────
            inf_t0 = time.time()
            try:
                results = self.model.track(
                    frame,
                    persist=True,
                    tracker=self.tracker_cfg,
                    conf=self.conf_thres,
                    iou=self.iou_thres,
                    classes=self.filter_indices if self.filter_indices else None,
                    verbose=False,
                    **device_kw,
                )
                self.monitor.record("inference", (time.time() - inf_t0) * 1000.0)
            except Exception as e:
                print(f"⚠ Inference error: {e}")
                continue

            # ── Draw annotations (task-specific) ──────────────────────────
            draw_t0   = time.time()
            annotated = frame.copy()

            if self.model_task == 'pose':
                num_obj = self._draw_pose(annotated, results)
            elif self.model_task == 'segment':
                num_obj = self._draw_masks(annotated, results)
            else:
                # Default: detection / tracking
                num_obj = self._draw_detections(annotated, results)

            self.monitor.record("draw", (time.time() - draw_t0) * 1000.0)

            with self.frame_lock:
                self.latest_frame = annotated

            self.monitor.record("total", (time.time() - loop_t0) * 1000.0)

            # ── FPS counter (updated every second) ─────────────────────────
            fps_cnt += 1
            if time.time() - fps_tick >= 1.0:
                self.monitor.update_fps(fps_cnt)
                fps_cnt  = 0
                fps_tick = time.time()

            if frame_cnt % 100 == 0:
                s = self.monitor.get_stats()
                print(
                    f"[{frame_cnt:6d}] FPS={s['fps']:5.1f} | "
                    f"Inf={s['inference']['avg']:6.1f}ms | "
                    f"Obj={num_obj} | Clients={s['client_count']}"
                )

        self.loader.stop()
        self.stop_encode_event.set()
        if self.encode_thread and self.encode_thread.is_alive():
            self.encode_thread.join(timeout=2.0)
        self.monitor.print_report()

    # ──────────────────────────────────────────────────────────────────────────
    def run(self):
        """Start the HTTP server and the processing loop; block until stopped."""
        server = None

        def _sigint(signum, frame):
            print("\n🛑 Stopping …")
            self.running = False
            if server:
                server.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, _sigint)

        try:
            self.running = True

            proc_thr = threading.Thread(target=self.process_loop, daemon=True)
            proc_thr.start()

            server = ThreadedHTTPServer(("0.0.0.0", self.port), StreamingHandler)
            server.tracker = self

            srv_thr = threading.Thread(target=server.serve_forever, daemon=True)
            srv_thr.start()

            print(f"\n🌐 Streaming  → http://0.0.0.0:{self.port}")
            print(f"📊 Stats JSON → http://0.0.0.0:{self.port}/stats")
            print("\nPress Ctrl+C to terminate …\n")

            while self.running:
                time.sleep(0.5)

        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if server:
                server.shutdown()
            self.running = False
            print("\n🧹 Cleanup complete")


# ────────────────────────────────────────────────────────────────────────────
# CLI argument parser
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "YOLO HTTP-MJPEG Tracker v1.1.0 (Model-Driven Backend + Full-Task)\n\n"
            "Backend is selected automatically from the model file:\n"
            "  *.pt                → PyTorch  (Ultralytics auto-device)\n"
            "  *.engine            → TensorRT (needs --device cuda or leave blank)\n"
            "  *_openvino_model/   → OpenVINO (default device: intel:cpu)\n"
            "  *.mlpackage         → CoreML   (macOS only)\n"
            "  *.onnx              → ONNX Runtime\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",   required=True,
                        help="RTSP URL or camera index (e.g. 0)")
    parser.add_argument("--model",   default="yolo26s.pt",
                        help="Model path: *.pt / *.engine / folder / *.onnx / *.mlpackage")
    parser.add_argument("--tracker", default="botsort.yaml",
                        help="Tracking config (botsort.yaml / bytetrack.yaml)")
    parser.add_argument("--port",    type=int, default=8000,
                        help="HTTP streaming port")
    parser.add_argument("--conf",    type=float, default=0.3,
                        help="Detection confidence threshold")
    parser.add_argument("--iou",     type=float, default=0.5,
                        help="IoU threshold for NMS")
    parser.add_argument(
        "--device", default="",
        help=(
            "Inference device. Leave empty for auto-detection.\n"
            "Examples: cuda, cuda:0, cpu, mps, xpu,\n"
            "          intel:cpu, intel:gpu, intel:npu\n"
            "The value is forwarded verbatim to Ultralytics — no pre-validation."
        ),
    )
    parser.add_argument("--frame-skip",        type=int, default=1,
                        help="Process every Nth frame (1 = every frame)")
    parser.add_argument("--show-id",           action="store_true",
                        help="Overlay object tracking IDs on labels")
    parser.add_argument("--trajectory",        action="store_true",
                        help="Draw object movement trajectory trails")
    parser.add_argument("--trajectory-length", type=int, default=30,
                        help="Maximum trajectory history points per object")
    parser.add_argument("--quality",           type=int, default=60,
                        help="JPEG encoding quality (0–100)")
    parser.add_argument("--classes",           nargs="+", type=str,
                        help="Filter detections to these class names, e.g. --classes person car")
    return parser.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Allow passing webcam index as a plain integer string
    if isinstance(args.input, str) and args.input.isdigit():
        args.input = int(args.input)

    tracker = YOLOTracker(args)
    tracker.run()


if __name__ == "__main__":
    main()
