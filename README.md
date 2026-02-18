# YOLO Environment Checker & HTTP Tracker  v1.0.0

A two-tool pipeline for deploying YOLO object detection/tracking on any hardware:

1. **`yolo_env_checker.py`** — Scan your environment, pick the optimal export format, and export the model.
2. **`yolo_http_tracker.py`** — Load the exported model and stream live annotated video over HTTP (MJPEG).

---

## Table of Contents

- [How the Two Tools Work Together](#how-the-two-tools-work-together)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Tool 1 — Environment Checker & Exporter](#tool-1--environment-checker--exporter)
- [Tool 2 — HTTP Tracker](#tool-2--http-tracker)
- [Supported Model Formats](#supported-model-formats)
- [Precision Reference](#precision-reference)
- [Pros & Cons](#pros--cons)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## How the Two Tools Work Together

```
┌─────────────────────────────┐       ┌──────────────────────────────────┐
│   yolo_env_checker.py       │       │   yolo_http_tracker.py           │
│   (Step 1 — Setup)          │──────▶│   (Step 2 — Runtime)             │
│                             │       │                                  │
│  • Scan CPU / GPU / RAM     │       │  • Load exported model           │
│  • Detect frameworks        │       │  • Auto-detect backend from      │
│  • Show precision matrix    │       │    file extension                │
│  • Interactive export menu  │       │  • Read RTSP / webcam            │
│                             │       │  • Run YOLO track per frame      │
│  Outputs:                   │       │  • Serve MJPEG stream on HTTP    │
│    yolo26s_fp16.engine      │       │  • Live stats (JSON + HTML)      │
│    yolo26s_fp16.onnx        │       │                                  │
│    yolo26s_fp16_openvino/   │       │                                  │
└─────────────────────────────┘       └──────────────────────────────────┘
```

**Typical workflow:**

```bash
# Step 1 — check environment and export
python yolo_env_checker.py
# → follow the interactive menu → exports e.g. yolo26s_fp16.engine

# Step 2 — run the tracker
python yolo_http_tracker.py \
    --input rtsp://192.168.1.100/stream \
    --model yolo26s_fp16.engine
# → open http://localhost:8000 in a browser
```

---

## Features

### Environment Checker (`yolo_env_checker.py`)

| Feature | Detail |
|---|---|
| CPU detection | Model name, physical/logical cores, RAM |
| AI ISA extensions | FMA3, AVX, AVX2, AVX512 family, AMX-BF16/INT8, NEON, SVE |
| Windows ISA (4 sources) | NumPy → py-cpuinfo → Kernel32 → PowerShell .NET (union) |
| OS GPU list | Windows (WMI), macOS (system_profiler), Linux (lspci / /sys) |
| NVIDIA details | nvidia-smi driver version, CUDA version, VRAM per GPU |
| PyTorch check | CUDA / MPS / XPU availability, version mismatch diagnosis |
| Precision matrix | FP32/FP16/BF16/INT8/INT4/FP8 — verified by actual tensor allocation |
| Framework detection | TensorRT, CoreML, OpenVINO |
| Smart diagnostics | AMD/Intel GPU suggestions, version mismatch fixes |
| Interactive export | Format → model → precision menus with recommended defaults |

### HTTP Tracker (`yolo_http_tracker.py`)

| Feature | Detail |
|---|---|
| Model-driven backend | Backend inferred from file extension; no manual flag needed |
| Multi-client MJPEG | Unlimited simultaneous browser/VLC viewers |
| HTML viewer | Auto-refresh stats panel (FPS, inference ms, drop rate, clients) |
| JSON stats endpoint | `GET /stats` — suitable for external dashboards |
| Corner-box overlay | Cleaner than full rectangles; less frame occlusion |
| Trajectory trails | Fading per-object paths (`--trajectory`) |
| Tracking ID labels | Show `#id class conf%` overlays (`--show-id`) |
| Independent JPEG encoder | Dedicated thread — encoding never blocks inference |
| Active frame dropping | Stale frames discarded before inference; drop counter tracked |
| OS-aware RTSP backends | Windows: FFMPEG/MSMF/DSHOW; Linux: FFMPEG/GStreamer/V4L2 |
| Auto-reconnect | Seamless stream recovery on network interruption |
| Class filtering | Restrict detection to specified class names (`--classes`) |
| Frame skip | Process every Nth frame for lower-spec hardware (`--frame-skip`) |
| Idle pause | Processing pauses when no client is connected (saves resources) |

---

## Requirements

### Minimum

```
Python >= 3.8
PyTorch >= 2.0
ultralytics
opencv-python
numpy
```

### Optional (for specific backends)

| Package | Purpose |
|---|---|
| `tensorrt` | TensorRT engine inference (NVIDIA GPUs) |
| `coremltools` | CoreML inference (macOS / Apple Silicon) |
| `openvino` | OpenVINO inference (Intel CPUs/GPUs, AMD GPUs) |
| `cpuinfo` (py-cpuinfo) | Enhanced CPU ISA detection on Windows |

---

## Installation

### 1. Clone and enter the repo

```bash
git clone https://github.com/your-username/yolo-export-and-stream.git
cd yolo-export-and-stream
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
```

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source ./venv/bin/activate
```

```bash
pip install --upgrade pip
```

### 3. Install dependencies — pick your platform

#### macOS (CPU / Apple Silicon MPS)

```bash
pip install torch torchvision torchaudio
pip install ultralytics
```

#### Linux — CPU only

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python-headless
```

> Use `opencv-python-headless` on headless servers (no display required).

#### Linux — NVIDIA GPU (CUDA)

```bash
pip install torch torchvision torchaudio
pip install ultralytics
```

#### Linux — NVIDIA GPU + TensorRT

```bash
pip install torch torchvision torchaudio
pip install ultralytics tensorrt
```

### 4. Optional packages

```bash
pip install coremltools   # CoreML export (macOS only)
pip install openvino      # OpenVINO export/inference (Intel/AMD)
pip install py-cpuinfo    # Enhanced CPU ISA detection on Windows
```

---

### CUDA Toolkit (required for TensorRT export)

If you plan to export TensorRT `.engine` files, the CUDA Toolkit (nvcc) must
be installed and match your PyTorch CUDA build.

**Check current version:**

```bash
nvcc --version
```

**Download:** https://developer.nvidia.com/cuda-downloads

**Update PyTorch to match your CUDA version** (example: CUDA 13.0):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

**Update system CUDA on Linux** (example: CUDA 13.1.1):

```bash
wget https://developer.download.nvidia.com/compute/cuda/13.1.1/local_installers/cuda_13.1.1_590.48.01_linux.run
sudo sh cuda_13.1.1_590.48.01_linux.run
```

> After updating system CUDA, also reinstall PyTorch with the matching
> `--index-url` as shown above so both versions align.

---

> **AMD GPU on Linux:** Use the ROCm build of PyTorch — https://rocm.docs.amd.com/  
> **AMD GPU on Windows:** PyTorch does not support ROCm on Windows; use OpenVINO or ONNX Runtime instead.

---

## Quick Start

### Check environment and export a model

```bash
python yolo_env_checker.py
```

The script will:
1. Print a full environment report (CPU, GPU, precision matrix, frameworks)
2. Ask you to select an export format, model size, and precision
3. Export the model and print the usage snippet

### Stream from a webcam

```bash
python yolo_http_tracker.py --input 0 --model yolo26s.pt
```

### Stream from an IP camera

```bash
python yolo_http_tracker.py \
    --input rtsp://admin:pass@192.168.1.100/stream \
    --model yolo26s_fp16.engine \
    --conf 0.35 \
    --show-id \
    --trajectory
```

Open **http://localhost:8000** in any browser or VLC.

---

## Tool 1 — Environment Checker & Exporter

### Running

```bash
python yolo_env_checker.py
```

### Output sections

```
🔥 YOLO Environment Checker & Model Export Tool  v1.0.0
  📦 Ultralytics
  📋 System Information
  🧮 CPU Information
  🖥️  GPU Hardware List (OS-level detection)
  🎮 NVIDIA GPU (nvidia-smi level)
  🔥 PyTorch (framework level)
  ⚡ CUDA Toolkit
  📊 Precision Support Matrix
  🚀 Inference Acceleration Frameworks
  🩺 Smart Diagnostics & Recommendations
```

### Customising export parameters

Edit the constants at the top of the file:

```python
EXPORT_IMGSZ     = 640     # Input size
TENSORRT_DYNAMIC = False   # Dynamic batch for TensorRT
TENSORRT_WORKSPACE = 4     # GB allocated for TRT engine build
ONNX_SIMPLIFY    = True    # Simplify ONNX graph
CALIBRATION_DATA = 'coco8.yaml'  # INT8 calibration dataset
```

---

## Tool 2 — HTTP Tracker

### Basic usage

```bash
python yolo_http_tracker.py --input <source> --model <model_path> [options]
```

### All options

| Option | Default | Description |
|---|---|---|
| `--input` | *required* | RTSP URL or webcam index (e.g. `0`) |
| `--model` | `yolo26s.pt` | Model path — format auto-detected from extension |
| `--tracker` | `botsort.yaml` | Tracking algorithm config |
| `--port` | `8000` | HTTP server port |
| `--conf` | `0.3` | Detection confidence threshold |
| `--iou` | `0.5` | IoU threshold for NMS |
| `--device` | *(auto)* | Override inference device (e.g. `cuda`, `cpu`, `mps`, `intel:gpu`) |
| `--frame-skip` | `1` | Process every Nth frame |
| `--show-id` | off | Show tracking IDs on labels |
| `--trajectory` | off | Draw movement trails |
| `--trajectory-length` | `30` | Max history points per object |
| `--quality` | `60` | JPEG encoding quality (0–100) |
| `--classes` | *(all)* | Filter class names, e.g. `--classes person car` |

### HTTP endpoints

| Path | Response | Use |
|---|---|---|
| `/` | HTML | Browser viewer with live stats |
| `/stream` | MJPEG | Direct stream (VLC, ffplay, img tag) |
| `/stats` | JSON | Performance metrics for dashboards |

---

## Supported Model Formats

| File / path | Backend | Notes |
|---|---|---|
| `*.pt` / `*.yaml` | PyTorch | Ultralytics auto-selects device |
| `*.engine` | TensorRT | CUDA required; fastest on NVIDIA |
| `*_openvino_model/` | OpenVINO | Best for Intel CPUs, GPUs, NPUs |
| `*.mlpackage` / `*.mlmodel` | CoreML | macOS / Apple Silicon only |
| `*.onnx` | ONNX Runtime | Universal; good cross-platform choice |

---

## Precision Reference

| Precision | Who benefits |
|---|---|
| FP32 | Baseline; works everywhere |
| FP16 | NVIDIA Pascal+ / Apple M-series — ~2× speed, minimal accuracy loss |
| BF16 | NVIDIA Ampere+ (hardware); older cards emulate it |
| INT8 | NVIDIA Turing+ / Intel VNNI — ~4× speed; requires calibration dataset |
| INT4 | TensorRT 8+ on Turing+ — highest throughput, most accuracy loss |
| FP8 | NVIDIA Ada Lovelace / Hopper (CC ≥8.9) |

---

## Pros & Cons

### Pros

- **Zero configuration for most cases** — backend and device are inferred automatically from the model file; no flags needed.
- **Hardware-honest precision matrix** — uses actual PyTorch tensor allocation rather than static lookup tables.
- **Works across all major platforms** — NVIDIA CUDA, Apple MPS, Intel XPU/OpenVINO, AMD (via OpenVINO/ONNX).
- **Multi-client capable** — any number of browsers or VLC instances can view the stream simultaneously.
- **Resource-efficient** — processing pauses when no client is connected; stale frames are dropped rather than queued.
- **Resilient streaming** — automatic RTSP reconnection; per-backend fallback list for OpenCV capture.
- **Windows-friendly ISA detection** — four independent CPU feature sources reconciled via union.
- **Clean visual output** — corner-box style (less occlusion), semi-transparent labels, fading trajectory trails.

### Cons

- **No built-in authentication** — the HTTP stream is open; do not expose port 8000 to untrusted networks without a reverse proxy.
- **Single camera per process** — to track multiple streams simultaneously you need to run multiple instances.
- **MJPEG latency** — introduces a few hundred milliseconds of end-to-end latency; not suitable for sub-100 ms real-time control loops.
- **TensorRT build time** — first export to `.engine` can take 10–30 minutes depending on GPU and model size.
- **INT8 calibration required** — INT8 TensorRT/OpenVINO exports need a representative calibration dataset (`coco8.yaml` by default).
- **CoreML macOS-only** — the CoreML path only works on macOS; there is no Windows/Linux CoreML equivalent.
- **AMD GPU on Windows** — PyTorch does not include ROCm for Windows; AMD hardware is accessible only through OpenVINO or ONNX Runtime DirectML.
- **No GPU-side frame decode** — video decoding is CPU-bound (OpenCV); hardware video decode is not integrated.
- **No recording / playback** — the tool streams live only; recording to disk must be done externally (e.g. ffmpeg pipe).

---

## Troubleshooting

### CUDA is detected but PyTorch cannot use it

```
⚠️ System has NVIDIA GPU but PyTorch cannot access CUDA!
```

Check that PyTorch CUDA version matches the installed driver:

```bash
# Check driver CUDA version
nvidia-smi | grep "CUDA Version"

# Reinstall PyTorch for the correct CUDA version (e.g. 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### TensorRT warm-up fails

Make sure you are using the same CUDA version that was used to build the engine:

```bash
python -c "import tensorrt; print(tensorrt.__version__)"
nvcc --version
```

### OpenVINO device string

Use Ultralytics' device parameter (passed via `--device`):

```bash
--device intel:cpu    # CPU inference
--device intel:gpu    # iGPU / dGPU (Arc)
--device intel:npu    # Neural Processing Unit (Meteor Lake+)
```

### Stream does not open (RTSP)

1. Test with `ffplay rtsp://...` to confirm the URL works.
2. On Windows, try specifying the backend explicitly:
   ```bash
   # Nothing is needed — the loader tries all backends automatically
   ```
3. If behind a NAT, try adding `?transport=tcp` to the RTSP URL.

### AMD GPU not used by PyTorch (Windows)

PyTorch for Windows does not include ROCm. Use OpenVINO instead:

```bash
pip install openvino
python yolo_env_checker.py   # export to OpenVINO format
python yolo_http_tracker.py --model yolo26s_fp16_openvino_model --device intel:gpu
```

---

## Project Structure

```
yolo-env-tracker/
├── yolo_env_checker.py     # Step 1: environment check + model export
├── yolo_http_tracker.py    # Step 2: live HTTP streaming tracker
└── README.md
```

---

## License

MIT License — free to use, modify, and distribute.
