# YOLO 環境檢測工具 & HTTP 串流追蹤器

一套兩件式流水線，可在任何硬體上部署 YOLO 物件偵測、姿勢估計與實例分割：

1. **`yolo_env_checker.py`** — 掃描硬體環境、選擇最佳匯出格式並匯出模型。
2. **`yolo_http_tracker.py`** — 載入匯出的模型，透過 HTTP（MJPEG）即時串流標註影像。

支援任務一覽（v1.1.0）：

| 任務 | 模型後綴 | 畫面呈現 |
|---|---|---|
| 物件偵測 / 追蹤 | `yolo26?.pt` | 角框 + 標籤 + 軌跡線 |
| 姿勢估計 | `yolo26?-pose.pt` | COCO 17 關節點骨架，按肢體區域分色 |
| 實例分割 | `yolo26?-seg.pt` | 半透明逐實例彩色遮罩 + 角框 |

---

## 目錄

- [兩個工具如何協作](#兩個工具如何協作)
- [功能特色](#功能特色)
- [系統需求](#系統需求)
- [安裝方式](#安裝方式)
- [快速開始](#快速開始)
- [工具一：環境檢測與模型匯出](#工具一環境檢測與模型匯出)
- [工具二：HTTP 串流追蹤器](#工具二http-串流追蹤器)
- [支援的任務](#支援的任務)
- [支援的模型格式](#支援的模型格式)
- [精度對照表](#精度對照表)
- [優點與限制](#優點與限制)
- [疑難排解](#疑難排解)
- [專案結構](#專案結構)
- [授權](#授權)
- [第三方聲明](#第三方聲明)

---

## 兩個工具如何協作

```
┌─────────────────────────────┐       ┌──────────────────────────────────┐
│   yolo_env_checker.py       │       │   yolo_http_tracker.py           │
│   （第一步：設定）           │──────▶│   （第二步：執行）               │
│                             │       │                                  │
│  • 掃描 CPU / GPU / RAM     │       │  • 載入匯出的模型                │
│  • 偵測推理框架              │       │  • 從副檔名自動判斷後端           │
│  • 顯示精度支援矩陣          │       │  • 讀取 RTSP / 網路攝影機        │
│  • 互動式匯出選單            │       │  • 對每幀執行 YOLO 追蹤          │
│                             │       │  • HTTP 提供 MJPEG 串流          │
│  輸出範例：                  │       │  • 即時效能統計（JSON + HTML）   │
│    yolo26s_fp16.engine      │       │                                  │
│    yolo26s_fp16.onnx        │       │                                  │
│    yolo26s_fp16_openvino/   │       │                                  │
└─────────────────────────────┘       └──────────────────────────────────┘
```

**典型使用流程：**

```bash
# 第一步：檢測環境並匯出模型
python yolo_env_checker.py
# → 依照互動選單操作 → 匯出例如 yolo26s_fp16.engine

# 第二步：啟動追蹤器
python yolo_http_tracker.py \
    --input rtsp://192.168.1.100/stream \
    --model yolo26s_fp16.engine
# → 在瀏覽器開啟 http://localhost:8000
```

---

## 功能特色

### 環境檢測工具（`yolo_env_checker.py`）

| 功能 | 說明 |
|---|---|
| CPU 偵測 | 型號、實體核心 / 邏輯執行緒、記憶體容量 |
| AI ISA 指令集擴充 | FMA3、AVX、AVX2、AVX512 系列、AMX-BF16/INT8、NEON、SVE |
| Windows ISA（四來源） | NumPy → py-cpuinfo → Kernel32 → PowerShell .NET（取聯集） |
| OS 層 GPU 清單 | Windows (WMI)、macOS (system_profiler)、Linux (lspci / /sys) |
| NVIDIA GPU 詳情 | nvidia-smi 驅動版本、CUDA 版本、各卡 VRAM |
| PyTorch 診斷 | CUDA / MPS / XPU 可用性、版本不匹配自動診斷 |
| 精度支援矩陣 | FP32/FP16/BF16/INT8/INT4/FP8，以實際 tensor 分配驗證 |
| 推理框架偵測 | TensorRT、CoreML、OpenVINO |
| 智慧診斷 | AMD / Intel GPU 建議、版本衝突修復方案 |
| 互動式選單 | 格式 → 任務 → 模型大小 → 精度，附推薦預設值 |

### HTTP 串流追蹤器（`yolo_http_tracker.py`）

| 功能 | 說明 |
|---|---|
| 模型驅動後端 | 從副檔名自動推斷後端，無需手動指定 |
| 模型驅動任務 | 從檔名後綴（`-pose`、`-seg`）推斷任務，並於載入後以 `model.task` 確認 |
| 物件偵測 / 追蹤 | 角框覆蓋、標籤、信心值、可選軌跡線 |
| 姿勢估計 | COCO 17 關節點骨架，按肢體區域分色，低於 `--pose-kp-conf` 門檻的關節點自動跳過 |
| 實例分割 | 半透明逐實例彩色遮罩（20 色調色盤依類別循環），輪廓描邊，角框 + 標籤疊加 |
| 多客戶端 MJPEG | 支援不限數量的瀏覽器 / VLC 同時觀看 |
| HTML 檢視頁面 | 自動更新統計面板（任務、FPS、推理時間、丟幀率、連線數） |
| JSON 統計端點 | `GET /stats`，適合外部儀表板串接 |
| 角框樣式 | 比完整矩形更簡潔，減少畫面遮擋 |
| 軌跡線 | 逐物件淡出路徑（`--trajectory`），三種任務皆支援 |
| 追蹤 ID 標籤 | `--show-id` 顯示 `#id 類別 信心值%` |
| 獨立 JPEG 編碼執行緒 | 專用執行緒，編碼不阻塞推理 |
| 主動丟幀機制 | 過舊幀在推理前丟棄，並計入丟幀計數器 |
| 感知 OS 的 RTSP 後端 | Windows：FFMPEG/MSMF/DSHOW；Linux：FFMPEG/GStreamer/V4L2 |
| 自動重連 | 串流中斷後無縫恢復 |
| 類別過濾 | 透過 `--classes` 限制偵測的類別名稱 |
| 幀跳過 | `--frame-skip` 讓低規硬體也能流暢執行 |
| 閒置暫停 | 無客戶端連線時暫停處理以節省資源 |

---

## 系統需求

### 最低需求

```
Python >= 3.8
PyTorch >= 2.0
ultralytics
opencv-python
numpy
```

### 選用套件（依後端需求）

| 套件 | 用途 |
|---|---|
| `tensorrt` | TensorRT 引擎推理（NVIDIA GPU） |
| `coremltools` | CoreML 推理（macOS / Apple Silicon） |
| `openvino` | OpenVINO 推理（Intel CPU/GPU、AMD GPU） |
| `cpuinfo`（py-cpuinfo） | Windows 上增強 CPU ISA 偵測精度 |

---

## 安裝方式

### 1. 複製專案並進入目錄

```bash
git clone https://github.com/patrick3399/yolo26-export-and-stream.git
cd yolo26-export-and-stream
```

### 2. 建立並啟用虛擬環境

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

### 3. 依平台安裝相依套件

#### macOS（CPU / Apple Silicon MPS）

```bash
pip install torch torchvision torchaudio
pip install ultralytics
```

#### Linux — 僅 CPU

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python-headless
```

> 無顯示介面的伺服器請使用 `opencv-python-headless`。

#### Linux — NVIDIA GPU（CUDA）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install ultralytics
```

#### Linux — NVIDIA GPU + TensorRT

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install ultralytics tensorrt
```

### 4. 選用套件

```bash
pip install coremltools   # CoreML 匯出（僅限 macOS）
pip install openvino      # OpenVINO 匯出 / 推理（Intel / AMD）
pip install py-cpuinfo    # Windows 上增強 CPU ISA 偵測
```

---

### CUDA Toolkit（TensorRT 匯出必要）

若需匯出 TensorRT `.engine` 檔，系統必須安裝 CUDA Toolkit（nvcc），且版本需與 PyTorch 的 CUDA 編譯版本一致。

**確認目前版本：**

```bash
nvcc --version
```

**下載：** https://developer.nvidia.com/cuda-downloads

**更新 PyTorch 以匹配您的 CUDA 版本**（範例：CUDA 13.0）：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

**在 Linux 上更新系統 CUDA**（範例：CUDA 13.1.1）：

```bash
wget https://developer.download.nvidia.com/compute/cuda/13.1.1/local_installers/cuda_13.1.1_590.48.01_linux.run
sudo sh cuda_13.1.1_590.48.01_linux.run
```

> 更新系統 CUDA 後，請同步以對應的 `--index-url` 重新安裝 PyTorch，確保版本一致。

---

> **Linux AMD GPU：** 請使用 ROCm 版 PyTorch — https://rocm.docs.amd.com/  
> **Windows AMD GPU：** PyTorch 不支援 Windows 上的 ROCm，請改用 OpenVINO 或 ONNX Runtime。

---

## 快速開始

### 第一步：檢測環境（並選擇性地匯出模型）

```bash
python yolo_env_checker.py
```

腳本將依序：
1. 印出完整環境報告（CPU、GPU、精度矩陣、推理框架）
2. 要求選擇**任務**（偵測 / 姿勢 / 分割）
3. 要求選擇模型大小與匯出格式 / 精度
4. 執行匯出並印出使用範例

> **PyTorch 格式：無需匯出步驟**  
> 若選擇 **PyTorch** 格式（CUDA / MPS / XPU / CPU），工具不會產生任何檔案，
> 直接將 `.pt` 模型名稱傳給 `yolo_http_tracker.py` 即可。  
> 若本地沒有該檔案，**Ultralytics 會在首次執行時自動下載**。
>
> ```bash
> # 若 yolo26s.pt 不存在，Ultralytics 會自動下載
> python yolo_http_tracker.py --input 0 --model yolo26s.pt
> ```

### 第二步：物件偵測 / 追蹤（網路攝影機）

```bash
python yolo_http_tracker.py --input 0 --model yolo26s.pt --show-id --trajectory
```

### 第二步：姿勢估計（IP 攝影機）

```bash
python yolo_http_tracker.py \
    --input rtsp://admin:pass@192.168.1.100/stream \
    --model yolo26s-pose.pt \
    --show-id --trajectory
```

### 第二步：實例分割（已匯出的 engine）

```bash
python yolo_http_tracker.py \
    --input rtsp://admin:pass@192.168.1.100/stream \
    --model yolo26s-seg_fp16.engine \
    --conf 0.35
```

在任意瀏覽器或 VLC 開啟 **http://localhost:8000**。  
資訊面板的 **Task** 欄位會顯示目前執行的任務模式。

---

## 工具一：環境檢測與模型匯出

### 執行方式

```bash
python yolo_env_checker.py
```

### 報告各節說明

```
🔥 YOLO Environment Checker & Model Export Tool  v1.1.0
  📦 Ultralytics
  📋 系統資訊
  🧮 CPU 資訊
  🖥️  GPU 硬體清單（OS 層偵測）
  🎮 NVIDIA GPU（nvidia-smi 層）
  🔥 PyTorch（框架層）
  ⚡ CUDA Toolkit
  📊 精度支援矩陣
  🚀 推理加速框架
  🩺 智慧診斷與建議
```

### 互動選單流程（v1.1.0）

```
第一層：選擇匯出格式
  PyTorch (CUDA / MPS / XPU / CPU) | TensorRT | CoreML | OpenVINO | ONNX

第二層：選擇任務
  [1] 物件偵測 / 追蹤    yolo26?.pt
  [2] 姿勢估計           yolo26?-pose.pt
  [3] 實例分割           yolo26?-seg.pt

第三層：選擇模型大小
  [1] Nano  [2] Small  [3] Medium  [4] Large  [5] XLarge
  （或直接輸入自訂檔名）

第四層：選擇精度（PyTorch 格式跳過此步驟）
  FP32 | FP16 ⭐ | INT8
```

工具會自動組合模型檔名（例如 `yolo26s-pose.pt`）並傳遞給匯出步驟。  
三種任務的匯出參數完全相同，任務資訊已編碼在模型權重中，而非匯出旗標。

### PyTorch 格式：不產生匯出檔案

選擇 **PyTorch** 格式（CUDA / MPS / XPU / CPU）時，工具僅印出使用範例後結束，不寫入任何檔案。
這是刻意設計的行為——PyTorch 模型由 Ultralytics 直接從 `.pt` 權重載入。

若本地沒有對應的 `.pt` 檔，Ultralytics 會在 `yolo_http_tracker.py` 啟動時自動下載：

```bash
# 若 yolo26s-pose.pt 不存在，Ultralytics 會自動下載
python yolo_http_tracker.py --input 0 --model yolo26s-pose.pt
```

### 自訂匯出參數

編輯腳本頂部的常數：

```python
EXPORT_IMGSZ       = 640     # 輸入圖像尺寸
TENSORRT_DYNAMIC   = False   # TensorRT 動態 batch
TENSORRT_WORKSPACE = 4       # TRT 引擎建置分配的 GB 數
ONNX_SIMPLIFY      = True    # 簡化 ONNX 計算圖
CALIBRATION_DATA   = 'coco8.yaml'  # INT8 校準資料集
```

---

## 工具二：HTTP 串流追蹤器

### 基本用法

```bash
python yolo_http_tracker.py --input <來源> --model <模型路徑> [選項]
```

### 所有選項

| 選項 | 預設值 | 說明 |
|---|---|---|
| `--input` | **必填** | RTSP URL 或攝影機索引（如 `0`） |
| `--model` | `yolo26s.pt` | 模型路徑，格式從副檔名自動偵測 |
| `--tracker` | `botsort.yaml` | 追蹤演算法設定檔 |
| `--port` | `8000` | HTTP 伺服器埠號 |
| `--conf` | `0.3` | 偵測信心值門檻 |
| `--iou` | `0.5` | NMS 的 IoU 門檻 |
| `--device` | *（自動）* | 覆寫推理裝置（如 `cuda`、`cpu`、`mps`、`intel:gpu`） |
| `--frame-skip` | `1` | 每 N 幀處理一次 |
| `--show-id` | 關閉 | 在標籤上顯示追蹤 ID |
| `--trajectory` | 關閉 | 繪製移動軌跡線 |
| `--trajectory-length` | `30` | 每個物件的最大軌跡歷史點數 |
| `--quality` | `60` | JPEG 編碼品質（0–100） |
| `--classes` | *（全部）* | 過濾類別名稱，如 `--classes person car` |
| `--pose-kp-conf` | `0.3` | 姿勢估計的關節點信心值門檻；低於此值的關節點不予繪製 |

### HTTP 端點

| 路徑 | 回應類型 | 用途 |
|---|---|---|
| `/` | HTML | 瀏覽器檢視頁面，含即時統計 |
| `/stream` | MJPEG | 直接串流（VLC、ffplay、img 標籤） |
| `/stats` | JSON | 效能指標，適合外部儀表板 |

---

## 支援的任務

任務從模型檔名自動偵測，載入後以 `model.task` 確認，**無需** `--task` 旗標。

| 任務 | 模型檔名規則 | 判斷依據 |
|---|---|---|
| 物件偵測 / 追蹤 | `yolo26?.pt`、`yolo26?.engine` 等 | 預設值 |
| 姿勢估計 | 檔名含 `*-pose*` | `-pose` 後綴 |
| 實例分割 | 檔名含 `*-seg*` | `-seg` 後綴 |

### 物件偵測 / 追蹤

標準 YOLO 物件偵測，搭配多目標追蹤（BotSort / ByteTrack）。

- 角框樣式邊界框（比完整矩形遮擋更少）
- 半透明標籤顯示類別名稱、信心值，可選追蹤 ID
- 可選每個追蹤物件的淡出軌跡線（`--trajectory`）
- 透過 `--classes person car` 等過濾類別

### 姿勢估計

偵測畫面中的人物並估計每人 17 個 COCO 身體關節點。

- 骨架肢體依身體區域分色繪製：
  - **黃色** — 臉部連線
  - **青色** — 左臂（肩 → 肘 → 腕）
  - **紅粉色** — 右臂
  - **灰色** — 軀幹（肩髖連線、髖部橫桿）
  - **綠色** — 左腿
  - **藍色** — 右腿
- 信心值低於 `--pose-kp-conf` 門檻（預設 `0.3`）的關節點自動跳過（不會出現幽靈肢體）
- 邊界框與標籤疊加在骨架上方
- 透過追蹤 ID 支援軌跡線（`--trajectory`）

> **注意：** 姿勢模型僅偵測 `person` 類別，`--classes` 過濾不適用。

### 實例分割

為每個偵測到的物件產生逐像素實例遮罩。

- 20 色調色盤依類別 ID 循環（同類別跨幀顏色一致）
- 遮罩以 40% 透明度渲染，場景背景仍清晰可見
- 輪廓描邊以 100% 不透明度繪製，邊緣銳利
- 角框與標籤疊加在遮罩上方
- 支援全部 80 個 COCO 類別（或自訂訓練模型的類別）
- 透過追蹤 ID 支援軌跡線（`--trajectory`）

---

## 支援的模型格式

| 檔案 / 路徑 | 後端 | 備註 |
|---|---|---|
| `*.pt` / `*.pth` / `*.yaml` | PyTorch | Ultralytics 自動選擇裝置 |
| `*.engine` | TensorRT | 需要 CUDA；NVIDIA 上速度最快 |
| `*_openvino_model/` | OpenVINO | 最適合 Intel CPU、GPU、NPU |
| `*.mlpackage` / `*.mlmodel` | CoreML | 僅限 macOS / Apple Silicon |
| `*.onnx` | ONNX Runtime | 通用格式，跨平台相容性最佳 |

---

## 精度對照表

| 精度 | 適用對象 |
|---|---|
| FP32 | 基準線，所有硬體皆支援 |
| FP16 | NVIDIA Pascal+ / Apple M 系列 — 約 2× 速度，精度損失極小 |
| BF16 | NVIDIA Ampere+（硬體原生）；較舊的顯示卡以模擬方式支援 |
| INT8 | NVIDIA Turing+ / Intel VNNI — 約 4× 速度，需要校準資料集 |
| INT4 | TensorRT 8+ on Turing+ — 最高吞吐量，精度損失最大 |
| FP8 | NVIDIA Ada Lovelace / Hopper（CC ≥8.9） |

---

## 優點與限制

### 優點

- **多數情況無需設定** — 後端與裝置從模型檔自動推斷，無需手動指定旗標。
- **真實的精度矩陣** — 以實際 PyTorch tensor 分配驗證，而非靜態查表。
- **跨平台支援** — NVIDIA CUDA、Apple MPS、Intel XPU/OpenVINO、AMD（透過 OpenVINO/ONNX）。
- **多客戶端串流** — 任意數量的瀏覽器或 VLC 可同時觀看。
- **資源節省** — 無客戶端連線時暫停處理；過舊幀直接丟棄而非排隊等待。
- **穩健的串流恢復** — 自動 RTSP 重連；OpenCV 讀流有多後端備援清單。
- **Windows 友善的 ISA 偵測** — 四個獨立 CPU 特性來源，取聯集確保準確。
- **簡潔的視覺輸出** — 角框樣式（減少遮擋）、半透明標籤、淡出軌跡線。

### 限制

- **無內建身分驗證** — HTTP 串流為開放存取，請勿直接將 8000 埠暴露於不信任網路，應搭配反向代理。
- **每個 Process 僅支援單一攝影機** — 若需同時追蹤多路串流，需啟動多個 Process。
- **MJPEG 延遲** — 端對端延遲約數百毫秒，不適用於需要 100ms 以下回應的即時控制系統。
- **TensorRT 引擎建置時間長** — 首次匯出 `.engine` 依 GPU 與模型大小可能需 10–30 分鐘。
- **INT8 需校準資料集** — TensorRT / OpenVINO 的 INT8 匯出需要代表性資料集（預設 `coco8.yaml`）。
- **CoreML 僅限 macOS** — CoreML 路徑僅在 macOS 上可用，Windows / Linux 無法使用。
- **Windows 上 AMD GPU 限制** — Windows 版 PyTorch 不含 ROCm，AMD 硬體只能透過 OpenVINO 或 ONNX Runtime DirectML 存取。
- **無 GPU 端影片解碼** — 影片解碼由 CPU（OpenCV）負責，未整合硬體加速解碼。
- **無錄製 / 回放功能** — 本工具僅提供即時串流，錄製需透過外部工具（如 ffmpeg pipe）。

---

## 疑難排解

### 偵測到 CUDA 但 PyTorch 無法使用

```
⚠️ System has NVIDIA GPU but PyTorch cannot access CUDA!
```

確認 PyTorch CUDA 版本與已安裝的驅動程式版本一致：

```bash
# 確認驅動的 CUDA 版本
nvidia-smi | grep "CUDA Version"

# 重新安裝對應 CUDA 版本的 PyTorch（以 CUDA 12.1 為例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### TensorRT 熱身失敗

確認使用的 CUDA 版本與建置引擎時相同：

```bash
python -c "import tensorrt; print(tensorrt.__version__)"
nvcc --version
```

### OpenVINO 裝置字串

透過 `--device` 傳遞 Ultralytics 的裝置參數：

```bash
--device intel:cpu    # CPU 推理
--device intel:gpu    # 內顯 / 獨顯（Arc 系列）
--device intel:npu    # 神經處理單元（Meteor Lake 以上）
```

### RTSP 串流無法開啟

1. 用 `ffplay rtsp://...` 確認 URL 可正常存取。
2. Windows 上無需手動指定後端，讀流器會自動嘗試所有後端。
3. 若在 NAT 後方，試著在 RTSP URL 後加上 `?transport=tcp`。

### Windows 上 AMD GPU 未被 PyTorch 使用

Windows 版 PyTorch 不含 ROCm，請改用 OpenVINO：

```bash
pip install openvino
python yolo_env_checker.py   # 匯出為 OpenVINO 格式
python yolo_http_tracker.py --model yolo26s_fp16_openvino_model --device intel:gpu
```

---

## 專案結構

```
yolo26-export-and-stream/
├── yolo_env_checker.py     # 第一步：環境檢測 + 模型匯出（v1.1.0）
├── yolo_http_tracker.py    # 第二步：即時 HTTP 串流追蹤器（v1.1.0）
├── LICENSE
├── README.md
└── README.zh.md
```

---

## 授權

本專案腳本（`yolo_env_checker.py`、`yolo_http_tracker.py`）以
**GNU Affero General Public License v3.0（AGPLv3）** 發布。

依據 AGPLv3，您可以自由使用、修改與散布本軟體，但須遵守以下條件：

- 若您修改本軟體並透過網路提供服務（例如架設 SaaS 或公開 HTTP 端點），  
  **必須**以 AGPLv3 開放您的修改後原始碼。
- 散布時必須包含原始授權聲明及版權標示。
- 衍生作品必須採用相同授權條款。

完整授權條款：[License](./LICENSE)  

---

## 第三方聲明

### Ultralytics 與 YOLO26

> © 2026 Ultralytics Inc. All rights reserved.

本專案依賴 [Ultralytics](https://github.com/ultralytics/ultralytics) 函式庫，
並使用 YOLO26 模型權重，兩者均為 Ultralytics Inc. 的智慧財產，
受其各自授權條款約束。

- Ultralytics 授權：https://github.com/ultralytics/ultralytics/blob/main/LICENSE
- 商業使用 Ultralytics 模型可能需要額外授權，詳見：https://www.ultralytics.com/license

本專案為獨立工具，透過 Ultralytics 公開 API 運作，與 Ultralytics Inc.
無隸屬、背書或官方支援關係。
