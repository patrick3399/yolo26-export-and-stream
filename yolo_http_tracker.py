#!/usr/bin/env python3
"""
YOLO HTTP-MJPEG Tracker  v1.0.0  (Model-Driven Backend Edition)

Overview:
    Step 2 in the two-tool pipeline.
    Takes a YOLO model exported by yolo_env_checker.py (or any supported
    format) and streams annotated video over HTTP as an MJPEG feed.

Backend selection:
    The inference backend is determined automatically from the model path:
        *.pt / *.pth / *.yaml   → PyTorch  (Ultralytics auto-device)
        *.engine                → TensorRT (CUDA required)
        *_openvino_model/       → OpenVINO (folder)
        *.mlpackage / *.mlmodel → CoreML   (macOS only)
        *.onnx                  → ONNX Runtime

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
        print("📊 PERFORMANCE REPORT (v1.0.0 Model-Driven Backend)")
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
        else:
            self.backend_list = [
                ("CAP_FFMPEG",    cv2.CAP_FFMPEG),
                ("CAP_GSTREAMER", cv2.CAP_GSTREAMER),
                ("CAP_V4L2",      cv2.CAP_V4L2),
            ]
            print(f"🐧 Linux/Unix — backends: {[b[0] for b in self.backend_list]}")

        self.cap = None
        self.connected = False
        self.backend_name = None

        self._open_capture()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _open_capture(self):
        """Try each backend in order; raise RuntimeError if all fail."""
        for name, code in self.backend_list:
            try:
                print(f"   Trying {name} ...", end="", flush=True)
                cap = cv2.VideoCapture(self.src, code)
                if code == cv2.CAP_FFMPEG:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Minimise latency
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
            ret, frame = self.cap.read()
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
  <title>YOLO Tracker v1.0.0</title>
  <style>
    body {{ margin:0; background:#000; color:#fff; font-family:Arial,sans-serif; }}
    .info {{
      position:absolute; top:10px; left:10px;
      background:rgba(0,0,0,.65); padding:10px 14px;
      border-radius:6px; font-size:14px; line-height:1.7;
    }}
    .fps {{ color:#00ff88; font-weight:bold; }}
    .fmt {{ color:#7dd3fc; }}
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
    YOLO HTTP-MJPEG Tracker  v1.0.0

    Combines model-format detection, RTSP/webcam reading, YOLO tracking,
    and HTTP serving into a single pipeline.

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
        print("🔍 SYSTEM DIAGNOSTIC (v1.0.0 Model-Driven Backend)")
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
        """Load the YOLO model and run a lightweight warm-up pass."""
        print(f"📦 Loading model: {self.model_name}  [{self.format_display}]")

        self.model = YOLO(self.model_name)

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
                        time.sleep(0.01)
                        continue
                    frame = self.latest_frame.copy()
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
        """
        w, h = x2 - x1, y2 - y1
        ll = max(15, int(min(w, h) * 0.18))
        # Top-left
        cv2.line(img, (x1, y1), (x1 + ll, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + ll), color, thickness)
        # Top-right
        cv2.line(img, (x2, y1), (x2 - ll, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + ll), color, thickness)
        # Bottom-right
        cv2.line(img, (x2, y2), (x2 - ll, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - ll), color, thickness)
        # Bottom-left
        cv2.line(img, (x1, y2), (x1 + ll, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - ll), color, thickness)

    def _draw_trajectory(self, img, track_id, center):
        """Draw a fading trajectory trail for a tracked object."""
        self.trajectories[track_id].append(center)
        pts = list(self.trajectories[track_id])
        if len(pts) < 2:
            return
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            cv2.line(
                img,
                tuple(map(int, pts[i - 1])),
                tuple(map(int, pts[i])),
                (0, 255, 0),
                max(1, int(2 * alpha)),
            )
        cv2.circle(img, tuple(map(int, center)), 4, (0, 255, 0), -1)

    def _draw_detections(self, img, results) -> int:
        """Annotate the frame with boxes, labels, and optional trajectories."""
        if not results or not results[0].boxes:
            return 0
        num = 0
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

            font  = cv2.FONT_HERSHEY_SIMPLEX
            scale, thick = 0.8, 2
            (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
            # Semi-transparent label background
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1 - th - 12), (x1 + tw + 12, y1), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            cv2.putText(img, label, (x1 + 6, y1 - 6), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
            num += 1
        return num

    # ──────────────────────────────────────────────────────────────────────────
    def process_loop(self):
        """Main inference loop: read → infer → draw → update shared JPEG."""
        print("\n" + "=" * 70)
        print("✔ YOLO processing started  (v1.0.0 Model-Driven Backend)")
        print("=" * 70)
        print("Press Ctrl+C to stop and view performance report\n")

        self.loader = RTSPStreamLoader(self.input_src, monitor=self.monitor)

        device_kw     = self._build_device_kwargs()
        last_results  = None
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

                if results and results[0].boxes and len(results[0].boxes) > 0:
                    last_results = results
            except Exception as e:
                print(f"⚠ Inference error: {e}")
                continue

            # ── Draw annotations ───────────────────────────────────────────
            draw_t0  = time.time()
            annotated = frame.copy()
            num_obj  = self._draw_detections(annotated, last_results)
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
            "YOLO HTTP-MJPEG Tracker v1.0.0 (Model-Driven Backend)\n\n"
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
