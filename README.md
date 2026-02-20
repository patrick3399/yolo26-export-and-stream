# YOLO Environment Checker & HTTP Tracker

A two-tool pipeline for deploying YOLO object detection, pose estimation, and
segmentation on any hardware:

1. **`yolo_env_checker.py`** — Scan your environment, pick the optimal export format, and export the model.
2. **`yolo_http_tracker.py`** — Load the exported model and stream live annotated video over HTTP (MJPEG).

Supported tasks (v1.1.0):

| Task | Model suffix | What you see |
|---|---|---|
| Detection / Tracking | `yolo26?.pt` | Corner-box + label + trajectory trails |
| Pose Estimation | `yolo26?-pose.pt` | 17-keypoint COCO skeleton, colour-coded by limb zone |
| Segmentation | `yolo26?-seg.pt` | Semi-transparent per-instance colour masks + corner-box |

---

## Table of Contents

- [How the Two Tools Work Together](#how-the-two-tools-work-together)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Tool 1 — Environment Checker & Exporter](#tool-1--environment-checker--exporter)
- [Tool 2 — HTTP Tracker](#tool-2--http-tracker)
- [Supported Tasks](#supported-tasks)
- [Supported Model Formats](#supported-model-formats)
- [Precision Reference](#precision-reference)
- [Pros & Cons](#pros--cons)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [License](#license)
- [Third-Party Notices](#third-party-notices)

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
| Model-driven task | Task inferred from filename suffix (`-pose`, `-seg`); confirmed from `model.task` after load |
| Detection / Tracking | Corner-box overlay, label, confidence %, optional trajectory trails |
| Pose Estimation | COCO 17-keypoint skeleton; colour-coded zones (face / arms / legs); keypoints below `--pose-kp-conf` threshold skipped |
| Segmentation | Semi-transparent per-instance colour masks (20-colour palette by class); contour outline; corner-box + label on top |
| Multi-client MJPEG | Unlimited simultaneous browser/VLC viewers |
| HTML viewer | Auto-refresh stats panel (Task, FPS, inference ms, drop rate, clients) |
| JSON stats endpoint | `GET /stats` — suitable for external dashboards |
| Corner-box overlay | Cleaner than full rectangles; less frame occlusion |
| Trajectory trails | Fading per-object paths (`--trajectory`) — works across all tasks |
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
git clone https://github.com/patrick3399/yolo26-export-and-stream.git
cd yolo26-export-and-stream
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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install ultralytics
```

#### Linux — NVIDIA GPU + TensorRT

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
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

### Step 1 — Check environment (and optionally export)

```bash
python yolo_env_checker.py
```

The script will:
1. Print a full environment report (CPU, GPU, precision matrix, frameworks)
2. Ask you to select a **task** (Detection / Pose / Segmentation)
3. Ask you to select model size and export format / precision
4. Export the model and print the usage snippet

> **PyTorch format — no export needed:**  
> If you select a **PyTorch** format in the menu (PyTorch CUDA / MPS / XPU / CPU),
> no file is exported. Simply pass the `.pt` model name directly to
> `yolo_http_tracker.py`. If the file does not exist locally, **Ultralytics will
> download it automatically** on first run.
>
> ```bash
> # No prior export step required — Ultralytics downloads yolo26s.pt if missing
> python yolo_http_tracker.py --input 0 --model yolo26s.pt
> ```

### Step 2 — Detection / Tracking (webcam)

```bash
python yolo_http_tracker.py --input 0 --model yolo26s.pt --show-id --trajectory
```

### Step 2 — Pose Estimation (IP camera)

```bash
python yolo_http_tracker.py \
    --input rtsp://admin:pass@192.168.1.100/stream \
    --model yolo26s-pose.pt \
    --show-id --trajectory
```

### Step 2 — Segmentation (exported engine)

```bash
python yolo_http_tracker.py \
    --input rtsp://admin:pass@192.168.1.100/stream \
    --model yolo26s-seg_fp16.engine \
    --conf 0.35
```

Open **http://localhost:8000** in any browser or VLC.  
The **Task** field in the info overlay confirms which mode is active.

---

## Tool 1 — Environment Checker & Exporter

### Running

```bash
python yolo_env_checker.py
```

### Output sections

```
🔥 YOLO Environment Checker & Model Export Tool  v1.1.0
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

### Interactive menu flow (v1.1.0)

```
Step 1 — Select export format
  PyTorch (CUDA / MPS / XPU / CPU) | TensorRT | CoreML | OpenVINO | ONNX

Step 2 — Select task
  [1] Detection / Tracking     yolo26?.pt
  [2] Pose Estimation          yolo26?-pose.pt
  [3] Segmentation             yolo26?-seg.pt

Step 3 — Select model size
  [1] Nano   [2] Small   [3] Medium   [4] Large   [5] XLarge
  (or type a custom filename directly)

Step 4 — Select precision  (skipped for PyTorch formats)
  FP32 | FP16 ⭐ | INT8
```

The composed model filename (e.g. `yolo26s-pose.pt`) is passed automatically
to the export step. All three task types export with identical parameters;
the task is encoded in the model weights, not the export flags.

### PyTorch format — no export file produced

When you select a **PyTorch** format (CUDA / MPS / XPU / CPU), the tool prints
a usage snippet and exits without writing any file. This is intentional —
PyTorch models are loaded directly from `.pt` weights by Ultralytics.

If the `.pt` file does not exist on disk, Ultralytics downloads it automatically
when `yolo_http_tracker.py` starts:

```bash
# yolo26s-pose.pt will be downloaded on first run if not present locally
python yolo_http_tracker.py --input 0 --model yolo26s-pose.pt
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
| `--pose-kp-conf` | `0.3` | Keypoint confidence threshold for pose estimation; keypoints below this value are not drawn |

### HTTP endpoints

| Path | Response | Use |
|---|---|---|
| `/` | HTML | Browser viewer with live stats |
| `/stream` | MJPEG | Direct stream (VLC, ffplay, img tag) |
| `/stats` | JSON | Performance metrics for dashboards |

---

## Supported Tasks

Task is detected automatically from the model filename and confirmed from
`model.task` after loading. No `--task` flag is needed.

| Task | Model filename pattern | Detection source |
|---|---|---|
| Detection / Tracking | `yolo26?.pt`, `yolo26?.engine`, etc. | default |
| Pose Estimation | `*-pose*` in filename | `-pose` suffix |
| Segmentation | `*-seg*` in filename | `-seg` suffix |

### Detection / Tracking

Standard YOLO object detection with multi-object tracking (BotSort / ByteTrack).

- Corner-box style bounding boxes (less occlusion than full rectangles)
- Semi-transparent label with class name, confidence, and optional track ID
- Optional fading trajectory trails per tracked object (`--trajectory`)
- Class filter via `--classes person car` etc.

### Pose Estimation

Detects people and estimates 17 COCO body keypoints per person.

- Skeleton limbs drawn with colour-coded body zones:
  - **Yellow** — face connections
  - **Cyan** — left arm (shoulder → elbow → wrist)
  - **Red/Pink** — right arm
  - **Grey** — torso (shoulder–hip connections, hip bar)
  - **Green** — left leg
  - **Blue** — right leg
- Keypoints below the `--pose-kp-conf` threshold (default `0.3`) are skipped (no phantom limbs)
- Bounding box and label drawn on top of the skeleton
- Trajectory trails supported via track ID (`--trajectory`)

> **Note:** Pose models only detect the `person` class.  
> `--classes` filtering is not applicable.

### Segmentation

Produces per-pixel instance masks for each detected object.

- 20-colour palette cycling by class ID (consistent colours across frames)
- Masks rendered at 40 % opacity so the scene remains visible
- Contour outline drawn at full opacity for a crisp edge
- Corner-box and label drawn on top of the mask
- All 80 COCO classes supported (or custom classes from your trained model)
- Trajectory trails supported via track ID (`--trajectory`)

---

## Supported Model Formats

| File / path | Backend | Notes |
|---|---|---|
| `*.pt` / `*.pth` / `*.yaml` | PyTorch | Ultralytics auto-selects device |
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
yolo26-export-and-stream/
├── yolo_env_checker.py     # Step 1: environment check + model export (v1.1.0)
├── yolo_http_tracker.py    # Step 2: live HTTP streaming tracker  (v1.1.0)
├── LICENSE
├── README.md
└── README.zh.md
```

---

## License

The scripts in this project (`yolo_env_checker.py`, `yolo_http_tracker.py`) are released under the **GNU Affero General Public License v3.0 (AGPLv3)**.

Under the AGPLv3, you are free to use, modify, and distribute this software, subject to the following conditions:

* If you modify this software and provide it as a service over a network (e.g., running it as SaaS or exposing public HTTP endpoints), you **must** make your modified source code available under the AGPLv3.
* You must include the original license statement and copyright notice when distributing.
* Derivative works must be licensed under the same terms.

Full license terms: [License](./LICENSE)  

---

## Third-Party Notices

### Ultralytics & YOLO26

> © 2026 Ultralytics Inc. All rights reserved.

This project depends on the [Ultralytics](https://github.com/ultralytics/ultralytics)
library and uses YOLO26 model weights, both of which are the intellectual
property of Ultralytics Inc. and are subject to their own license terms.

- Ultralytics license: https://github.com/ultralytics/ultralytics/blob/main/LICENSE
- Commercial use of Ultralytics models may require a separate license.  
  See https://www.ultralytics.com/license for details.

This project is an independent tool that interfaces with the Ultralytics API.
It is not affiliated with, endorsed by, or officially supported by Ultralytics Inc.
