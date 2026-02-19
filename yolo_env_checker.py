#!/usr/bin/env python3
"""
YOLO Environment Checker & Model Export Tool

Overview:
    Step 1 in the two-tool pipeline.
    Run this script first to inspect your hardware/software environment,
    then export a YOLO model to the format best suited to your system.
    The exported file is consumed by yolo_http_tracker.py (Step 2).

What it does:
    - Detects CPU model, core count, RAM, and AI-relevant ISA extensions
    - Detects GPU hardware via OS-level APIs (lspci / Win32 / system_profiler)
    - Detects NVIDIA GPU details via nvidia-smi
    - Detects PyTorch, CUDA, MPS (Apple Metal), XPU (Intel)
    - Tests actual precision support (FP32/FP16/BF16/INT8/INT4/FP8) per device
    - Detects inference frameworks: TensorRT, CoreML, OpenVINO
    - Provides an interactive menu to select task, model size, format, and precision
    - Exports the chosen model with the recommended parameters

Supported tasks (v1.1.0):
    Detection / Tracking  — yolo26n/s/m/l/x.pt
    Pose Estimation       — yolo26n/s/m/l/x-pose.pt
    Segmentation          — yolo26n/s/m/l/x-seg.pt

ISA detection scope:
    Kept  : FMA3, AVX, AVX2, AVX512 family, AMX (meaningful AI throughput delta)
    Removed: SSE2/SSE3/SSE4.x (universal on modern CPUs, no filtering value)

Windows ISA detection uses four independent sources (union):
    NumPy __cpu_features__  →  py-cpuinfo  →  Kernel32  →  PowerShell .NET
"""

import sys
import os
import platform
import subprocess
import warnings
import multiprocessing
import argparse
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*scikit-learn.*")
warnings.filterwarnings("ignore", message=".*openvino.runtime.*")
os.environ['PYTHONWARNINGS'] = 'ignore'


# ============================================================================
# Export parameter defaults  (edit these to customise export behaviour)
# All values follow the official Ultralytics export attribute documentation.
# ============================================================================

EXPORT_IMGSZ        = 640      # Input image size (Ultralytics default: 640)

# TensorRT  (supported: imgsz, half, dynamic, int8, workspace, batch, data)
TENSORRT_DYNAMIC    = False
TENSORRT_WORKSPACE  = 4        # GB; Ultralytics default: 4
TENSORRT_BATCH      = 1

# ONNX  (supported: imgsz, half, dynamic, simplify, opset, nms, batch, device)
# Note: ONNX does NOT support int8
ONNX_DYNAMIC        = False
ONNX_SIMPLIFY       = True     # Simplify via onnxslim; Ultralytics default: True
ONNX_OPSET          = None     # None = latest version
ONNX_NMS            = False
ONNX_BATCH          = 1

# OpenVINO  (supported: imgsz, half, int8, dynamic, nms, batch, data, fraction)
OPENVINO_DYNAMIC    = False
OPENVINO_NMS        = False
OPENVINO_BATCH      = 1
OPENVINO_FRACTION   = 1.0      # Fraction of calibration data used for INT8; default: 1.0

# CoreML  (supported: imgsz, half, int8, nms, batch, device; no dynamic)
COREML_NMS          = True     # Ultralytics default: False; recommended True for iOS/macOS
COREML_BATCH        = 1

# INT8 calibration dataset  (shared by TensorRT / OpenVINO / CoreML)
CALIBRATION_DATA    = 'coco8.yaml'

# ============================================================================
# Virtual / display-only GPU keywords
# These GPUs have no AI compute capability and are classified separately.
# ============================================================================
VIRTUAL_GPU_KEYWORDS = [
    'QXL', 'VMWARE', 'VIRTUALBOX', 'RED HAT', 'CIRRUS',
    'MICROSOFT BASIC DISPLAY', 'MICROSOFT RENDER', 'HYPER-V',
    'BOCHS', 'VBOX', 'VIRTIO', 'PARAVIRTUAL', 'BASIC RENDER',
    'STANDARD VGA', 'GENERIC VGA',
]
# ============================================================================


class YOLOEnvChecker:
    """YOLO Environment Checker & Model Export Tool  v1.1.0"""

    def __init__(self):
        self.env_info = {
            'ultralytics':  {},
            'system':       {},
            'cpu_detail':   {},
            'os_gpu':       {},
            'gpu_hardware': {},
            'pytorch':      {},
            'cuda':         {},
            'frameworks':   {},
            'precision':    [],
        }

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def print_header(self, text: str):
        print(f"\n{'='*70}")
        print(f"  {text}")
        print(f"{'='*70}\n")

    def print_section(self, text: str):
        print(f"\n{'-'*70}")
        print(f"  {text}")
        print(f"{'-'*70}\n")

    # ------------------------------------------------------------------
    # 1. Ultralytics
    # ------------------------------------------------------------------

    def detect_ultralytics(self):
        result = {'installed': False, 'version': None}
        try:
            from ultralytics import __version__
            result['installed'] = True
            result['version']   = __version__
        except Exception:
            pass
        self.env_info['ultralytics'] = result

    # ------------------------------------------------------------------
    # 2. System information
    # ------------------------------------------------------------------

    def detect_system(self):
        system  = platform.system()
        machine = platform.machine()
        arch_map = {
            'x86_64': 'x64', 'AMD64': 'x64',
            'i386': 'x86',   'i686': 'x86',
            'arm64': 'aarch64', 'aarch64': 'aarch64',
        }
        self.env_info['system'] = {
            'os':      system,
            'arch':    arch_map.get(machine, machine),
            'machine': machine,
            'python':  platform.python_version(),
        }

    # ------------------------------------------------------------------
    # 3. CPU details  (model / cores / threads / RAM / ISA extensions)
    # ------------------------------------------------------------------

    def _run_cmd(self, cmd, shell=False, timeout=5, env=None) -> str:
        try:
            out = subprocess.check_output(
                cmd, shell=shell, stderr=subprocess.DEVNULL,
                timeout=timeout, env=env
            )
            return out.decode(errors='ignore').strip()
        except Exception:
            return ''

    def _get_cpu_model(self, os_type: str, machine: str) -> str:
        try:
            if os_type == 'Linux':
                with open('/proc/cpuinfo', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if line.startswith('model name'):
                            return line.split(':', 1)[1].strip()
                # ARM fallback
                for key in ('Hardware', 'CPU implementer'):
                    with open('/proc/cpuinfo', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            if line.startswith(key):
                                return line.split(':', 1)[1].strip()
            elif os_type == 'Darwin':
                v = self._run_cmd(['sysctl', '-n', 'machdep.cpu.brand_string'])
                return v if v else platform.processor()
            elif os_type == 'Windows':
                raw = self._run_cmd(
                    ['powershell', '-NoProfile', '-Command',
                     '(Get-CimInstance Win32_Processor).Name'],
                    timeout=8
                )
                if raw:
                    return raw.split('\n')[0].strip()
                # wmic fallback
                raw2 = self._run_cmd(['wmic', 'cpu', 'get', 'Name', '/value'])
                for ln in raw2.split('\n'):
                    if ln.startswith('Name='):
                        return ln.split('=', 1)[1].strip()
        except Exception:
            pass
        return platform.processor() or 'Unknown'

    def _get_cpu_cores(self, os_type: str) -> Tuple[Optional[int], int]:
        logical = multiprocessing.cpu_count()
        physical = None
        try:
            if os_type == 'Linux':
                cores_seen = set()
                cur_phys, cur_core = '0', None
                with open('/proc/cpuinfo', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if line.startswith('physical id'):
                            cur_phys = line.split(':')[1].strip()
                        elif line.startswith('core id'):
                            cur_core = line.split(':')[1].strip()
                            cores_seen.add((cur_phys, cur_core))
                if cores_seen:
                    physical = len(cores_seen)
                else:
                    # Force English output to avoid locale-dependent labels
                    c_env = {**os.environ, 'LANG': 'C', 'LC_ALL': 'C'}
                    raw = self._run_cmd(['lscpu'], env=c_env)
                    cps = skt = None
                    for ln in raw.split('\n'):
                        ln_lo = ln.lower()
                        if 'core' in ln_lo and 'socket' in ln_lo:
                            try: cps = int(ln.split(':')[1].strip())
                            except: pass
                        elif 'socket' in ln_lo and 'core' not in ln_lo:
                            try: skt = int(ln.split(':')[1].strip())
                            except: pass
                    if cps and skt:
                        physical = cps * skt
            elif os_type == 'Darwin':
                v = self._run_cmd(['sysctl', '-n', 'hw.physicalcpu'])
                if v.isdigit():
                    physical = int(v)
            elif os_type == 'Windows':
                v = self._run_cmd(
                    ['powershell', '-NoProfile', '-Command',
                     '(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum'],
                    timeout=8
                )
                if v and v.split('\n')[0].strip().isdigit():
                    physical = int(v.split('\n')[0].strip())
        except Exception:
            pass
        return physical, logical

    def _get_system_ram(self, os_type: str) -> str:
        try:
            if os_type == 'Linux':
                with open('/proc/meminfo', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('MemTotal'):
                            kb = int(line.split()[1])
                            return f"{kb / 1024 / 1024:.1f} GB"
            elif os_type == 'Darwin':
                v = self._run_cmd(['sysctl', '-n', 'hw.memsize'])
                if v.isdigit():
                    return f"{int(v) / 1024**3:.1f} GB"
            elif os_type == 'Windows':
                v = self._run_cmd(
                    ['powershell', '-NoProfile', '-Command',
                     '(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory'],
                    timeout=8
                )
                v = v.split('\n')[0].strip()
                if v.isdigit():
                    return f"{int(v) / 1024**3:.1f} GB"
        except Exception:
            pass
        return '—'

    # ------------------------------------------------------------------
    # ISA detection — source 1: NumPy __cpu_features__
    # ------------------------------------------------------------------

    def _isa_from_numpy(self) -> List[str]:
        """
        Read CPU feature flags from NumPy's internal __cpu_features__ dict.
        Supports both NumPy 1.x (numpy.core) and 2.x (numpy._core).
        This is the most reliable Windows source because NumPy is always
        present in any YOLO environment.
        """
        result = []
        try:
            cpu_features = None
            try:
                from numpy._core import _multiarray_umath      # NumPy 2.x
                cpu_features = _multiarray_umath.__cpu_features__
            except (ImportError, AttributeError):
                from numpy.core import _multiarray_umath        # NumPy 1.x
                cpu_features = _multiarray_umath.__cpu_features__

            # Normalise internal names → display names
            NORM = {
                'FMA': 'FMA3',   'FMA3': 'FMA3',
                'AVX': 'AVX',    'AVX2': 'AVX2',
                'AVX512F': 'AVX512F',   'AVX512BW': 'AVX512BW',
                'AVX512VL': 'AVX512VL', 'AVX512VNNI': 'AVX512VNNI',
                'AVX512_BF16': 'AVX512_BF16', 'AVX512BF16': 'AVX512_BF16',
                'NEON': 'NEON',  'SVE': 'SVE',
                'AMX_BF16': 'AMX-BF16', 'AMX_INT8': 'AMX-INT8',
            }
            # Only extensions that meaningfully affect AI inference throughput
            # (excludes SSE2/3/4.x — universal on modern CPUs, not worth reporting)
            AI_KEYS = {
                'FMA3', 'AVX', 'AVX2',
                'AVX512F', 'AVX512BW', 'AVX512VL', 'AVX512VNNI', 'AVX512_BF16',
                'AMX-BF16', 'AMX-INT8',
                'NEON', 'SVE',
            }
            for feat, ok in cpu_features.items():
                if not ok:
                    continue
                norm = NORM.get(feat.upper().replace('-', '_'), feat.upper())
                if norm in AI_KEYS:
                    result.append(norm)
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------
    # ISA detection — source 2: py-cpuinfo
    # ------------------------------------------------------------------

    def _isa_from_cpuinfo(self) -> List[str]:
        """
        Read CPU flags from the py-cpuinfo library (optional dependency).
        Most complete source — can detect AMX and AVX512_BF16 details.
        """
        result = []
        try:
            import cpuinfo                               # type: ignore
            info  = cpuinfo.get_cpu_info()
            flags = [f.upper() for f in info.get('flags', [])]
            MAP = [
                ('FMA',        'FMA3'),    ('AVX',        'AVX'),
                ('AVX2',       'AVX2'),    ('AVX512F',    'AVX512F'),
                ('AVX512BW',   'AVX512BW'),('AVX512VL',   'AVX512VL'),
                ('AVX512VNNI', 'AVX512VNNI'),
                ('AVX512BF16', 'AVX512_BF16'),
                ('AMX_BF16',   'AMX-BF16'),('AMX_INT8',   'AMX-INT8'),
                ('NEON',       'NEON'),    ('SVE',        'SVE'),
            ]
            for flag, label in MAP:
                if flag in flags:
                    result.append(label)
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------
    # ISA detection — source 3: Windows Kernel32
    # ------------------------------------------------------------------

    def _isa_from_kernel32(self) -> List[str]:
        """
        Call Windows Kernel32 IsProcessorFeaturePresent (no external deps).
        PF_AVX=27, PF_AVX2=28, PF_AVX512F=31.
        Note: Windows API sometimes reports AVX2 without AVX; auto-corrected below.
        Ref: https://learn.microsoft.com/windows/win32/api/processthreadsapi
        """
        result = []
        try:
            import ctypes
            # Only query AVX-class features used by AI frameworks
            PF_MAP = {27: 'AVX', 28: 'AVX2', 31: 'AVX512F'}
            present = set()
            for fid, label in PF_MAP.items():
                if ctypes.windll.kernel32.IsProcessorFeaturePresent(fid):
                    present.add(label)
            if 'AVX2' in present:   # Fix known Windows API AVX reporting gap
                present.add('AVX')
            result = sorted(present)
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------
    # ISA detection — source 4: PowerShell .NET Runtime.Intrinsics
    # ------------------------------------------------------------------

    def _isa_from_powershell(self) -> List[str]:
        """
        Use PowerShell .NET Runtime.Intrinsics to detect FMA3 / VNNI etc.
        Slower to start than the other sources, but supplements FMA3/VNNI.
        Requires .NET SDK; silently skipped if unavailable.
        """
        result = []
        try:
            ps = (
                "$r=@();"
                "try{if([System.Runtime.Intrinsics.X86.Fma]::IsSupported){$r+='FMA3'}}catch{};"
                "try{if([System.Runtime.Intrinsics.X86.Avx]::IsSupported){$r+='AVX'}}catch{};"
                "try{if([System.Runtime.Intrinsics.X86.Avx2]::IsSupported){$r+='AVX2'}}catch{};"
                "try{if([System.Runtime.Intrinsics.X86.Avx512F]::IsSupported){$r+='AVX512F'}}catch{};"
                "try{if([System.Runtime.Intrinsics.X86.Avx512BW]::IsSupported){$r+='AVX512BW'}}catch{};"
                "try{if([System.Runtime.Intrinsics.X86.Avx512Vnni]::IsSupported){$r+='AVX512VNNI'}}catch{};"
                "$r -join ','"
            )
            raw = self._run_cmd(['powershell', '-NoProfile', '-Command', ps], timeout=10)
            if raw:
                result = [x.strip() for x in raw.split(',') if x.strip()]
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------
    # ISA detection — main entry point (platform-dispatched)
    # ------------------------------------------------------------------

    def _get_instruction_sets(self, os_type: str, machine: str) -> List[str]:
        """
        Detect CPU instruction-set extensions relevant to AI inference.

        Filter policy: only report extensions with a meaningful impact on
        AI workloads that are NOT universally present on pre-2010 hardware.
            Kept:    FMA3 (2013+), AVX (2011+), AVX2 (2013+),
                     AVX512F/BW/VL/VNNI/BF16 (2017+),
                     AMX-BF16/INT8 (Sapphire Rapids 2023+)
            Removed: SSE2/3/4.x (2000–2008, no filtering value today)

        Detection methods:
            Windows — union of four independent sources:
                      NumPy → py-cpuinfo → Kernel32 → PowerShell .NET
            Linux   — /proc/cpuinfo flags (most direct)
            macOS   — sysctl machdep.cpu.leaf7_features
            ARM     — /proc/cpuinfo features (Linux) or fixed list (Apple Silicon)
        """
        # ── ARM / Apple Silicon ──────────────────────────────────────────────
        if machine in ('arm64', 'aarch64'):
            found = []
            if os_type == 'Darwin':
                found.extend(['NEON', 'AMX', 'BF16(AMX)'])
            elif os_type == 'Linux':
                try:
                    with open('/proc/cpuinfo', encoding='utf-8', errors='ignore') as f:
                        content = f.read().lower()
                    if 'neon' in content or 'asimd' in content:
                        found.append('NEON')
                    if 'sve' in content:
                        found.append('SVE')
                    if 'bf16' in content:
                        found.append('BF16')
                except Exception:
                    pass
            return found

        # ── Linux (/proc/cpuinfo flags) ──────────────────────────────────────
        if os_type == 'Linux':
            found = []
            try:
                with open('/proc/cpuinfo', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        key = line.split(':')[0].strip().lower()
                        if key in ('flags', 'features'):
                            flags = set(line.split(':', 1)[1].lower().split())
                            MAP = [
                                ('fma',        'FMA3'),
                                ('avx',        'AVX'),
                                ('avx2',       'AVX2'),
                                ('avx512f',    'AVX512F'),
                                ('avx512bw',   'AVX512BW'),
                                ('avx512vl',   'AVX512VL'),
                                ('avx512vnni', 'AVX512VNNI'),
                                ('avx512bf16', 'AVX512_BF16'),
                                ('amx_bf16',   'AMX-BF16'),
                                ('amx_int8',   'AMX-INT8'),
                            ]
                            for flag, label in MAP:
                                if flag in flags:
                                    found.append(label)
                            break
            except Exception:
                pass
            return found

        # ── macOS (sysctl) ───────────────────────────────────────────────────
        if os_type == 'Darwin':
            found = []
            try:
                f1 = self._run_cmd(['sysctl', '-n', 'machdep.cpu.features']).upper()
                f2 = self._run_cmd(['sysctl', '-n', 'machdep.cpu.leaf7_features']).upper()
                combined = f1 + ' ' + f2
                MAP = [
                    ('FMA',        'FMA3'),   ('AVX1.0',    'AVX'),
                    ('AVX2.0',     'AVX2'),   ('AVX512F',   'AVX512F'),
                    ('AVX512BW',   'AVX512BW'),('AVX512VL', 'AVX512VL'),
                    ('AVX512VNNI', 'AVX512VNNI'),
                ]
                for flag, label in MAP:
                    if flag in combined:
                        found.append(label)
            except Exception:
                pass
            return found

        # ── Windows: union of four sources ───────────────────────────────────
        if os_type == 'Windows':
            merged: set = set()
            for src_fn in (
                self._isa_from_numpy,        # Source 1: NumPy (always present in YOLO env)
                self._isa_from_cpuinfo,      # Source 2: py-cpuinfo (most complete)
                self._isa_from_kernel32,     # Source 3: Kernel32 (no dependencies)
                self._isa_from_powershell,   # Source 4: PowerShell (supplements FMA3/VNNI)
            ):
                try:
                    merged.update(src_fn())
                except Exception:
                    pass

            ORDER = [
                'FMA3', 'AVX', 'AVX2',
                'AVX512F', 'AVX512BW', 'AVX512VL', 'AVX512VNNI', 'AVX512_BF16',
                'AMX-BF16', 'AMX-INT8',
            ]
            ordered   = [x for x in ORDER if x in merged]
            remaining = sorted(merged - set(ORDER))
            return ordered + remaining

        return []

    def detect_cpu_detail(self):
        os_type = platform.system()
        machine = platform.machine()
        physical, logical = self._get_cpu_cores(os_type)
        self.env_info['cpu_detail'] = {
            'model':    self._get_cpu_model(os_type, machine),
            'physical': physical,
            'logical':  logical,
            'ram':      self._get_system_ram(os_type),
            'isa':      self._get_instruction_sets(os_type, machine),
        }

    # ------------------------------------------------------------------
    # 4. OS-level GPU hardware detection
    # ------------------------------------------------------------------

    def detect_os_gpu_hardware(self):
        result = {'gpus': [], 'method': None, 'error': None}
        os_type = platform.system()
        try:
            if os_type == 'Windows':
                gpus = []
                flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                # Try PowerShell first
                try:
                    raw = subprocess.check_output(
                        ['powershell', '-NoProfile', '-Command',
                         'Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name'],
                        stderr=subprocess.STDOUT, creationflags=flags, timeout=10
                    ).decode(errors='ignore')
                    gpus = [ln.strip() for ln in raw.replace('\r\n', '\n').split('\n') if ln.strip()]
                    result['method'] = 'PowerShell Win32_VideoController'
                except Exception:
                    pass
                # Fallback to wmic if PowerShell failed or returned nothing
                if not gpus:
                    try:
                        raw = subprocess.check_output(
                            ['wmic', 'path', 'Win32_VideoController', 'get', 'Name', '/value'],
                            stderr=subprocess.DEVNULL, creationflags=flags, timeout=10
                        ).decode(errors='ignore')
                        gpus = [ln.split('=', 1)[1].strip()
                                for ln in raw.replace('\r\n', '\n').split('\n')
                                if ln.startswith('Name=')]
                        result['method'] = 'wmic Win32_VideoController'
                    except Exception:
                        pass

            elif os_type == 'Darwin':
                raw = self._run_cmd(
                    ['system_profiler', 'SPDisplaysDataType'], timeout=10
                )
                gpus = [ln.split(':', 1)[1].strip()
                        for ln in raw.split('\n')
                        if 'Chipset Model' in ln and ':' in ln]
                result['method'] = 'system_profiler SPDisplaysDataType'

            else:  # Linux
                try:
                    raw = subprocess.check_output(
                        ['lspci'], stderr=subprocess.DEVNULL, timeout=5
                    ).decode(errors='ignore')
                    gpus = []
                    for line in raw.split('\n'):
                        if any(k in line.upper() for k in ('VGA', '3D CONTROLLER', 'DISPLAY CONTROLLER')):
                            parts = line.split(':', 2)
                            name  = parts[-1].split('(rev')[0].strip() if len(parts) >= 3 else line.strip()
                            gpus.append(name)
                    result['method'] = 'lspci'
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    gpus = []
                    pci_path = '/sys/bus/pci/devices'
                    if os.path.isdir(pci_path):
                        for dev in os.listdir(pci_path):
                            cls_f = os.path.join(pci_path, dev, 'class')
                            if os.path.isfile(cls_f):
                                with open(cls_f) as f:
                                    if f.read().strip().startswith('0x03'):
                                        lbl = os.path.join(pci_path, dev, 'label')
                                        gpus.append(
                                            open(lbl).read().strip()
                                            if os.path.isfile(lbl) else f"PCI {dev}"
                                        )
                    result['method'] = '/sys/bus/pci/devices'

            result['gpus'] = gpus if gpus else ['No display adapter detected']
        except Exception as e:
            result['gpus']  = ['Detection failed']
            result['error'] = str(e)

        self.env_info['os_gpu'] = result

    # ------------------------------------------------------------------
    # 5. NVIDIA GPU details via nvidia-smi
    # ------------------------------------------------------------------

    def detect_gpu_hardware(self):
        result = {
            'has_nvidia_gpu':      False,
            'gpu_count':           0,
            'gpu_names':           [],
            'driver_version':      None,
            'driver_cuda_version': None,
            'gpu_details':         [],
        }
        try:
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                 '--format=csv,noheader'],
                stderr=subprocess.STDOUT, universal_newlines=True, timeout=10
            )
            lines = [ln.strip() for ln in out.strip().split('\n') if ln.strip()]
            result['has_nvidia_gpu'] = bool(lines)
            result['gpu_count']      = len(lines)
            for line in lines:
                p = [x.strip() for x in line.split(',')]
                if len(p) >= 2:
                    result['gpu_names'].append(p[0])
                    result['gpu_details'].append({'name': p[0], 'memory': p[1]})
                # Extract driver version from CSV (stable API)
                if len(p) >= 3 and p[2] and not result['driver_version']:
                    result['driver_version'] = p[2]

            # CUDA version is not available via --query-gpu; parse header
            try:
                smi = subprocess.check_output(
                    ['nvidia-smi'], stderr=subprocess.STDOUT,
                    universal_newlines=True, timeout=10
                )
                for ln in smi.split('\n'):
                    if 'CUDA Version:' in ln:
                        result['driver_cuda_version'] = ln.split('CUDA Version:')[1].strip().split()[0]
                        # Also grab driver version from header as fallback
                        if not result['driver_version'] and 'Driver Version:' in ln:
                            result['driver_version'] = ln.split('Driver Version:')[1].split('CUDA')[0].strip().split()[0]
                        break
            except Exception:
                pass
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
        except Exception:
            pass
        self.env_info['gpu_hardware'] = result

    # ------------------------------------------------------------------
    # 6. PyTorch
    # ------------------------------------------------------------------

    def detect_pytorch(self):
        result = {
            'installed': False, 'version': None,
            'cuda_available': False, 'cuda_version': None,
            'mps_available': False,
            'xpu_available': False, 'xpu_device_count': 0,
            'gpu_count': 0, 'gpu_names': [],
            'is_cpu_only': False,
            'cuda_error': None, 'cuda_init_error': None,
            'pytorch_cuda_version': None,
        }
        try:
            import torch
            result['installed'] = True
            result['version']   = torch.__version__
            if '+cpu' in torch.__version__:
                result['is_cpu_only'] = True
            if '+cu' in torch.__version__:
                cu = torch.__version__.split('+cu')[1].split('+')[0]
                result['pytorch_cuda_version'] = (
                    f"{cu[0:2]}.{cu[2]}" if len(cu) == 3 else
                    f"{cu[0]}.{cu[1]}"   if len(cu) == 2 else None
                )
            try:
                if torch.cuda.is_available():
                    result['cuda_available'] = True
                    result['cuda_version']   = torch.version.cuda
                    result['gpu_count']      = torch.cuda.device_count()
                    result['gpu_names']      = [
                        torch.cuda.get_device_name(i)
                        for i in range(result['gpu_count'])
                    ]
                else:
                    try:
                        _ = torch.cuda.device_count()
                    except Exception as e:
                        result['cuda_init_error'] = str(e)
            except Exception as e:
                result['cuda_error'] = result['cuda_init_error'] = str(e)

            if hasattr(torch.backends, 'mps'):
                result['mps_available'] = torch.backends.mps.is_available()
            if hasattr(torch, 'xpu'):
                try:
                    result['xpu_available']    = torch.xpu.is_available()
                    result['xpu_device_count'] = torch.xpu.device_count() if result['xpu_available'] else 0
                except Exception:
                    pass
        except ImportError:
            pass   # torch not installed; keep installed=False
        except Exception as e:
            result['cuda_error'] = str(e)
        self.env_info['pytorch'] = result

    # ------------------------------------------------------------------
    # 7. CUDA Toolkit (nvcc)
    # ------------------------------------------------------------------

    def detect_cuda_toolkit(self):
        result = {'installed': False, 'version': None, 'nvcc_warning': False, 'nvcc_warning_message': None}
        has_gpu = self.env_info.get('gpu_hardware', {}).get('has_nvidia_gpu', False)
        try:
            out = subprocess.check_output(['nvcc', '--version'],
                                          stderr=subprocess.STDOUT, universal_newlines=True, timeout=5)
            for ln in out.split('\n'):
                if 'release' in ln.lower():
                    result['version']   = ln.split('release')[1].split(',')[0].strip()
                    result['installed'] = True
                    break
        except FileNotFoundError:
            if has_gpu:
                result['nvcc_warning'] = True
                result['nvcc_warning_message'] = 'NVIDIA GPU detected but CUDA Toolkit is not installed (nvcc not found)'
        except Exception as e:
            if has_gpu:
                result['nvcc_warning'] = True
                result['nvcc_warning_message'] = f'nvcc error: {e}'
        self.env_info['cuda'] = result

    # ------------------------------------------------------------------
    # 8. Precision support (with actual tensor allocation test)
    # ------------------------------------------------------------------

    def _is_virtual_gpu(self, name: str) -> bool:
        up = name.upper()
        return any(k in up for k in VIRTUAL_GPU_KEYWORDS)

    def _get_apple_chip_name(self) -> str:
        raw = self._run_cmd(['sysctl', '-n', 'machdep.cpu.brand_string'])
        return raw if 'Apple' in raw else 'Apple Silicon'

    def _actual_precision_test(self, device_str: str, backend: str,
                               compute_cap: float = 0.0) -> dict:
        """
        Verify precision support by actually allocating torch.zeros on the device.
        INT8 / INT4 / FP8 are inference-framework features; inferred from compute capability.
        BF16: CUDA ≥8.0 = hardware native; else emulated (marked accordingly).
        """
        Y = '✅'
        N = '❌'
        res = {k: N for k in ('FP32', 'FP16', 'BF16', 'INT8', 'INT4', 'FP8')}
        try:
            import torch
        except ImportError:
            return {k: '—' for k in res}

        # FP32 — if device is completely unavailable, return immediately
        try:
            torch.zeros(1, device=device_str, dtype=torch.float32)
            res['FP32'] = Y
        except Exception:
            return res

        # FP16
        try:
            torch.zeros(1, device=device_str, dtype=torch.float16)
            res['FP16'] = Y
        except Exception:
            pass

        # BF16 — distinguish hardware native vs. emulated on CUDA
        if backend == 'CUDA':
            try:
                t = torch.zeros(1, device=device_str, dtype=torch.bfloat16)
                _ = t + t
                res['BF16'] = Y if compute_cap >= 8.0 else '⚠ emulated'
            except Exception:
                pass
        elif backend == 'MPS':
            res['BF16'] = '✅(M-series)'
        elif backend == 'XPU':
            res['BF16'] = '✅(Xe)'

        # INT8 / INT4 — inference-framework capability; based on compute capability
        if backend == 'CUDA':
            if compute_cap >= 7.5:
                res['INT8'] = Y
                res['INT4'] = '✅(TRT)'
            elif compute_cap >= 6.1:
                res['INT8'] = '✅(Pascal)'
        elif backend == 'MPS':
            res['INT8'] = '✅(CoreML)'
        elif backend == 'XPU':
            res['INT8'] = '⚠ (OV)'

        # FP8 — Ada Lovelace / Hopper+  (CC ≥8.9)
        if backend == 'CUDA' and compute_cap >= 8.9:
            res['FP8'] = Y

        return res

    def _build_os_only_precision(self, name: str, is_virtual: bool) -> dict:
        """
        Estimated precision for GPUs detected by the OS but unavailable via PyTorch.
        Virtual/display adapters return all dashes.
        """
        DASH = '—'
        base = {k: DASH for k in ('FP32', 'FP16', 'BF16', 'INT8', 'INT4', 'FP8')}
        if is_virtual:
            return base

        Y  = '✅'
        up = name.upper()

        if any(k in up for k in ('APPLE', 'M1', 'M2', 'M3', 'M4')):
            return {**base, 'FP32': Y, 'FP16': Y, 'BF16': Y, 'INT8': '✅(CoreML)'}

        elif any(k in up for k in ('RADEON', 'AMD', 'RX ', 'VEGA', 'NAVI', 'RDNA')):
            d = {**base, 'FP32': Y, 'FP16': '✅(DML)', 'INT8': '⚠ (ONNX)'}
            if any(k in up for k in ('RX 6', 'RX 7', '780M', '680M', 'RDNA2', 'RDNA3')):
                d['BF16'] = '⚠ (ROCm)'
            return d

        elif 'INTEL' in up:
            d = {**base, 'FP32': Y, 'FP16': '✅(OV)'}
            if any(k in up for k in ('ARC', 'XE', 'IRIS XE', 'A380', 'A770', 'B580', 'ULTRA', 'FLEX')):
                d.update({'BF16': '✅(OV)', 'INT8': '✅(OV)'})
            else:
                d['INT8'] = '⚠ (OV)'
            return d

        elif any(k in up for k in ('NVIDIA', 'GEFORCE', 'QUADRO', 'RTX', 'GTX', 'TESLA')):
            return {**base, 'FP32': Y, 'FP16': Y, 'INT8': '⚠ (Pascal+)'}

        return base

    def detect_precision_support(self):
        """
        Build the precision support matrix:
          - Safe when torch is not installed (no crash)
          - CUDA / MPS / XPU: actual tensor allocation test
          - OS-detected devices: estimated values (de-duplicated vs CUDA/MPS)
          - Virtual GPUs: separate category, no precision shown
        """
        os_g = self.env_info.get('os_gpu', {})
        pt   = self.env_info.get('pytorch', {})
        results: List[dict] = []

        torch_ok = False
        try:
            import torch
            torch_ok = True
        except ImportError:
            pass

        # ── CUDA devices (actual test) ──────────────────────────────────────
        if torch_ok and pt.get('cuda_available'):
            for i, name in enumerate(pt.get('gpu_names', [])):
                try:
                    props = torch.cuda.get_device_properties(i)
                    cc    = props.major + props.minor / 10
                    cc_str = f"{props.major}.{props.minor}"
                except Exception:
                    cc, cc_str = 0.0, 'unknown'
                results.append({
                    'category': 'pytorch',
                    'index':    i,
                    'name':     name,
                    'backend':  'CUDA',
                    'cc':       cc_str,
                    'matrix':   self._actual_precision_test(f'cuda:{i}', 'CUDA', cc),
                    'is_virtual': False,
                })

        # ── MPS device (actual test) ────────────────────────────────────────
        if torch_ok and pt.get('mps_available'):
            chip = self._get_apple_chip_name() if platform.system() == 'Darwin' else 'Apple Silicon'
            results.append({
                'category': 'pytorch',
                'index':    0,
                'name':     chip,
                'backend':  'MPS',
                'cc':       'N/A',
                'matrix':   self._actual_precision_test('mps', 'MPS'),
                'is_virtual': False,
            })

        # ── XPU device (actual test) ────────────────────────────────────────
        if torch_ok and pt.get('xpu_available'):
            for i in range(pt.get('xpu_device_count', 0)):
                try:
                    name = torch.xpu.get_device_name(i)
                except Exception:
                    name = f'Intel XPU {i}'
                results.append({
                    'category': 'pytorch',
                    'index':    i,
                    'name':     name,
                    'backend':  'XPU',
                    'cc':       'N/A',
                    'matrix':   self._actual_precision_test(f'xpu:{i}', 'XPU'),
                    'is_virtual': False,
                })

        # ── OS-detected devices (estimated, de-duplicated) ──────────────────
        cuda_names_lower = [n.lower() for n in pt.get('gpu_names', [])]
        mps_active = torch_ok and pt.get('mps_available', False)

        for gpu_name in os_g.get('gpus', []):
            if not gpu_name or gpu_name in ('No display adapter detected', 'Detection failed'):
                continue
            # Skip NVIDIA GPUs already covered by CUDA
            if pt.get('cuda_available') and 'NVIDIA' in gpu_name.upper():
                continue
            if pt.get('cuda_available') and any(
                n in gpu_name.lower() or gpu_name.lower() in n
                for n in cuda_names_lower
            ):
                continue
            # Skip Apple GPUs already covered by MPS
            if mps_active and 'APPLE' in gpu_name.upper():
                continue

            is_virt = self._is_virtual_gpu(gpu_name)
            results.append({
                'category':   'virtual' if is_virt else 'os_only',
                'index':      -1,
                'name':       gpu_name,
                'backend':    'Virtual display' if is_virt else 'OS detected',
                'cc':         'N/A',
                'matrix':     self._build_os_only_precision(gpu_name, is_virt),
                'is_virtual': is_virt,
            })

        # ── Fallback: no GPU found ───────────────────────────────────────────
        if not any(not r.get('is_virtual') for r in results):
            results.append({
                'category': 'none', 'index': -1,
                'name': 'No GPU detected', 'backend': '—', 'cc': 'N/A',
                'matrix': {k: '—' for k in ('FP32','FP16','BF16','INT8','INT4','FP8')},
                'is_virtual': False,
            })

        self.env_info['precision'] = results

    # ------------------------------------------------------------------
    # 9. Inference framework detection
    # ------------------------------------------------------------------

    def detect_tensorrt(self):
        try:
            import tensorrt as trt
            _ = trt.Builder(trt.Logger(trt.Logger.WARNING))
            return True, trt.__version__
        except Exception:
            return False, None

    def detect_coreml(self):
        if platform.system() != 'Darwin':
            return False, None
        try:
            import coremltools
            return True, coremltools.__version__
        except Exception:
            return False, None

    def detect_openvino(self):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                try:
                    import openvino as ov
                    _ = ov.Core()
                    return True, ov.__version__
                except Exception:
                    from openvino.runtime import Core
                    import openvino
                    _ = Core()
                    return True, openvino.__version__
        except Exception:
            return False, None

    # ------------------------------------------------------------------
    # 10. Full detection pipeline
    # ------------------------------------------------------------------

    def detect_all(self):
        self.print_header('🔍 Environment Check')
        print('Scanning environment, please wait...\n')
        self.detect_ultralytics()
        self.detect_system()
        self.detect_cpu_detail()
        self.detect_os_gpu_hardware()
        self.detect_gpu_hardware()
        self.detect_pytorch()
        self.detect_cuda_toolkit()
        self.detect_precision_support()

        trt_ok, trt_ver = self.detect_tensorrt()
        cml_ok, cml_ver = self.detect_coreml()
        ov_ok,  ov_ver  = self.detect_openvino()
        self.env_info['frameworks'] = {
            'tensorrt': {'available': trt_ok, 'version': trt_ver},
            'coreml':   {'available': cml_ok, 'version': cml_ver},
            'openvino': {'available': ov_ok,  'version': ov_ver},
        }

    # ------------------------------------------------------------------
    # 11. Environment report display
    # ------------------------------------------------------------------

    def _format_isa(self, isa: list) -> str:
        if not isa:
            return '(none detected)'
        indent = ' ' * 15
        lines, cur = [], ''
        for item in isa:
            candidate = f"{cur}  {item}" if cur else item
            if len(candidate) > 52:
                lines.append(cur)
                cur = item
            else:
                cur = candidate
        if cur:
            lines.append(cur)
        return f'\n{indent}'.join(lines)

    def _print_precision_block(self, items: list, title: str, note: str = ''):
        DTYPES = ['FP32', 'FP16', 'BF16', 'INT8', 'INT4', 'FP8']
        SEP    = '─' * 52
        print(f"  ▶ {title}")
        if note:
            print(f"    ⚠ {note}")
        for item in items:
            idx   = item['index']
            label = f"GPU {idx}" if idx >= 0 else 'GPU'
            print(f"  {SEP}")
            print(f"  {label} │ {item['name']}")
            print(f"  {'Backend':7} │ {item['backend']}  CC: {item['cc']}")
            print(f"  {SEP}")
            for dt in DTYPES:
                val = item['matrix'].get(dt, '—')
                print(f"  {dt:4} │ {val}")
        print(f"  {SEP}")

    def _print_virtual_list(self, items: list):
        print('  ▶ Virtual / display-only adapters (no AI compute capability)')
        for item in items:
            print(f"     ⊘ {item['name']}")

    def _print_cpu_compute(self):
        """CPU AI compute summary (independent of GPU precision matrix)."""
        pt    = self.env_info.get('pytorch', {})
        cpu_d = self.env_info.get('cpu_detail', {})
        isa   = cpu_d.get('isa', [])
        SEP   = '─' * 52
        print('  ▶ CPU Compute (independent of GPU)')
        print(f'  {SEP}')

        if pt.get('installed'):
            print('  CPU  │ ✅  PyTorch / ONNX Runtime CPU inference available')
        else:
            print('  CPU  │ ⊘  PyTorch not installed (ONNX Runtime CPU still usable)')

        has_amx     = any('AMX-INT8'  in s for s in isa)
        has_amx_bf  = any('AMX-BF16' in s for s in isa)
        has_vnni    = any('VNNI'      in s for s in isa)
        has_avx512  = any('AVX512F'   in s for s in isa)
        has_bf16    = any('AVX512_BF16' in s or 'BF16' in s for s in isa)
        has_avx2    = any('AVX2'      in s for s in isa)
        has_avx     = any(s == 'AVX'  for s in isa)
        has_fma3    = any('FMA3'      in s for s in isa)
        has_neon    = any('NEON'      in s or 'AMX' in s for s in isa)

        # INT8 inference capability
        if has_amx:
            print('  INT8 │ ✅  AMX-INT8 hardware acceleration (Intel Sapphire Rapids+)')
        elif has_vnni:
            print('  INT8 │ ✅  AVX512VNNI vector acceleration (Intel Cascade Lake+)')
        elif has_avx512:
            print('  INT8 │ ⚠   AVX512F (software quantization, no dedicated INT8 engine)')
        elif has_avx2:
            print('  INT8 │ ⚠   AVX2 (software quantization)')
        elif has_neon:
            print('  INT8 │ ✅  NEON / AMX (Apple Silicon)')
        else:
            print('  INT8 │ ❌  No SIMD acceleration')

        # BF16 / FP16 capability
        if has_amx_bf:
            print('  BF16 │ ✅  AMX-BF16 hardware acceleration')
        elif has_bf16:
            print('  BF16 │ ✅  AVX512_BF16 instruction set')
        elif has_neon and 'BF16(AMX)' in isa:
            print('  BF16 │ ✅  Apple Silicon AMX')

        # FP32 matrix performance hint
        if has_fma3:
            print('  FP32 │ ✅  FMA3 (fused multiply-add, faster matrix ops)')

        print(f'  {SEP}')

    def show_environment(self):
        ultra = self.env_info['ultralytics']
        sys_  = self.env_info['system']
        cpu_d = self.env_info['cpu_detail']
        os_g  = self.env_info['os_gpu']
        gpu_h = self.env_info['gpu_hardware']
        pt    = self.env_info['pytorch']
        cuda  = self.env_info['cuda']
        fw    = self.env_info['frameworks']
        prec  = self.env_info['precision']

        # ── Ultralytics ────────────────────────────────────────────────────
        self.print_section('📦 Ultralytics')
        if ultra['installed']:
            print(f"  ✅ Ultralytics {ultra['version']}")
        else:
            print('  ❌ Ultralytics not installed  →  pip install ultralytics')

        # ── System information ─────────────────────────────────────────────
        self.print_section('📋 System Information')
        print(f"  OS           : {sys_['os']}")
        print(f"  Architecture : {sys_['arch']} ({sys_['machine']})")
        print(f"  Python       : {sys_['python']}")

        # ── CPU details ───────────────────────────────────────────────────
        self.print_section('🧮 CPU Information')
        print(f"  CPU Model    : {cpu_d.get('model', '—')}")
        p, t = cpu_d.get('physical'), cpu_d.get('logical')
        cores_str = f"Physical {p} Core / Logical {t} Thread" if p else f"Logical {t} Thread"
        print(f"  Cores/Threads: {cores_str}")
        print(f"  System RAM   : {cpu_d.get('ram', '—')}")
        print(f"  AI ISA Exts  : {self._format_isa(cpu_d.get('isa', []))}")

        # ── OS-level GPU list ──────────────────────────────────────────────
        self.print_section('🖥️  GPU Hardware List (OS-level detection)')
        if os_g.get('error'):
            print(f"  ⚠️  Detection error: {os_g['error']}")
        else:
            print(f"  Method : {os_g.get('method', 'unknown')}")
        for i, name in enumerate(os_g.get('gpus', [])):
            flag = ' [virtual]' if self._is_virtual_gpu(name) else ''
            print(f"  [{i}] {name}{flag}")

        # ── NVIDIA GPU details (nvidia-smi) ───────────────────────────────
        self.print_section('🎮 NVIDIA GPU (nvidia-smi level)')
        if gpu_h.get('has_nvidia_gpu'):
            print(f"  ✅ Detected {gpu_h['gpu_count']} NVIDIA GPU(s)")
            for i, d in enumerate(gpu_h.get('gpu_details', [])):
                print(f"     GPU {i}: {d['name']} ({d['memory']})")
            drv = f"  ✅ Driver {gpu_h['driver_version']}"
            if gpu_h.get('driver_cuda_version'):
                drv += f"  (supports CUDA {gpu_h['driver_cuda_version']})"
            print(drv)
        else:
            print('  ⊘  No NVIDIA GPU detected (nvidia-smi not responding)')

        # ── PyTorch ────────────────────────────────────────────────────────
        self.print_section('🔥 PyTorch (framework level)')
        if pt.get('installed'):
            print(f"  ✅ PyTorch {pt['version']}")
            if pt.get('pytorch_cuda_version'):
                print(f"  🔌 PyTorch compiled CUDA: {pt['pytorch_cuda_version']}")

            if pt.get('cuda_available'):
                print(f"  ✅ CUDA available (version {pt['cuda_version']}, {pt['gpu_count']} GPU(s))")
                for i, n in enumerate(pt['gpu_names']):
                    print(f"     GPU {i}: {n}")
            else:
                has_nv = gpu_h.get('has_nvidia_gpu', False)
                if has_nv:
                    print('  ⚠️  System has NVIDIA GPU but PyTorch cannot access CUDA!')
                    pt_cu, dr_cu = pt.get('pytorch_cuda_version'), gpu_h.get('driver_cuda_version')
                    if pt_cu and dr_cu:
                        print(f'     PyTorch requires CUDA {pt_cu} / Driver supports {dr_cu}')
                        try:
                            if int(pt_cu.split('.')[0]) > int(dr_cu.split('.')[0]):
                                cu = pt_cu.replace('.', '')[:3]
                                print(f'  ❌ Version mismatch!')
                                print(f'     pip install torch torchvision torchaudio '
                                      f'--index-url https://download.pytorch.org/whl/cu{cu}')
                        except Exception:
                            pass
                    if pt.get('cuda_init_error'):
                        print(f'     CUDA error: {pt["cuda_init_error"]}')
                    print('     https://pytorch.org/get-started/locally/')
                elif pt.get('is_cpu_only'):
                    print('  ⊘  PyTorch CUDA: CPU-only build')
                else:
                    print('  ⊘  PyTorch CUDA: unavailable (no NVIDIA GPU)')

            if pt.get('mps_available'):
                print('  ✅ MPS (Apple Metal) available')
            elif sys_['os'] == 'Darwin':
                print('  ⊘  MPS: unavailable (requires macOS 12.3+ and Apple Silicon)')
            if pt.get('xpu_available'):
                print(f"  ✅ XPU (Intel) available ({pt['xpu_device_count']} device(s))")

        else:
            print('  ❌ PyTorch not installed  →  pip install torch')

        # ── CUDA Toolkit ───────────────────────────────────────────────────
        if cuda.get('installed'):
            self.print_section('⚡ CUDA Toolkit')
            print(f"  ✅ CUDA Toolkit {cuda['version']} (nvcc available)")
        elif cuda.get('nvcc_warning'):
            self.print_section('⚡ CUDA Toolkit')
            print(f"  ⚠️  {cuda['nvcc_warning_message']}")
            print('     https://developer.nvidia.com/cuda-toolkit')

        # ── Precision support matrix ───────────────────────────────────────
        self.print_section('📊 Precision Support Matrix')
        pytorch_items = [r for r in prec if r['category'] == 'pytorch']
        os_items      = [r for r in prec if r['category'] == 'os_only']
        virt_items    = [r for r in prec if r['category'] == 'virtual']
        none_items    = [r for r in prec if r['category'] == 'none']

        if pytorch_items:
            self._print_precision_block(pytorch_items, 'PyTorch-accessible devices (verified by test)')
        if os_items:
            print()
            self._print_precision_block(os_items, 'OS-detected devices (estimated)',
                                        'Not accessible via PyTorch; precision values are estimates')
        if none_items:
            print()
            self._print_precision_block(none_items, 'Device status')
        if virt_items:
            print()
            self._print_virtual_list(virt_items)
        print()
        self._print_cpu_compute()

        # ── Inference acceleration frameworks ─────────────────────────────
        self.print_section('🚀 Inference Acceleration Frameworks')
        if fw['tensorrt']['available']:
            print(f"  ✅ TensorRT {fw['tensorrt']['version']} (NVIDIA GPU acceleration)")
        else:
            print('  ⚠️  TensorRT not available')
            if pt.get('cuda_available'):
                print('     pip install tensorrt')

        if fw['coreml']['available']:
            print(f"  ✅ CoreML {fw['coreml']['version']} (Apple Silicon acceleration)")
        else:
            if sys_['os'] == 'Darwin':
                print('  ⚠️  CoreML not available  →  pip install coremltools')
            else:
                print('  ⊘  CoreML (macOS only)')

        if fw['openvino']['available']:
            print(f"  ✅ OpenVINO {fw['openvino']['version']} (Intel/AMD acceleration)")
        else:
            print('  ⚠️  OpenVINO not available  →  pip install openvino')

        # ── Smart diagnostics ─────────────────────────────────────────────
        self._show_smart_diagnostics()

    # ------------------------------------------------------------------
    # 12. Smart diagnostics
    # ------------------------------------------------------------------

    def _show_smart_diagnostics(self):
        os_g  = self.env_info.get('os_gpu', {})
        pt    = self.env_info.get('pytorch', {})
        fw    = self.env_info.get('frameworks', {})
        gpus  = os_g.get('gpus', [])

        has_amd    = any(k in g.upper() for g in gpus for k in ('AMD', 'RADEON', 'RX '))
        has_intel  = any('INTEL' in g.upper() for g in gpus)
        has_nvidia = any(k in g.upper() for g in gpus for k in ('NVIDIA', 'GEFORCE', 'RTX', 'GTX'))

        issues = []
        if has_amd and not pt.get('cuda_available') and not pt.get('xpu_available'):
            issues.append(('💡 AMD GPU detected but PyTorch cannot use it directly', [
                f'PyTorch {pt.get("version","?")} does not include AMD ROCm support (especially on Windows)',
                'Option 1: pip install onnxruntime  (DirectML can leverage AMD GPU)',
                'Option 2: pip install openvino      (OpenVINO supports AMD GPU)',
                'Option 3 (Linux): ROCm build of PyTorch → https://rocm.docs.amd.com/',
            ]))
        if has_intel and not pt.get('xpu_available') and not fw.get('openvino', {}).get('available'):
            issues.append(('💡 Intel GPU detected but acceleration not yet enabled', [
                'pip install openvino',
                'or XPU: pip install torch --index-url https://download.pytorch.org/whl/xpu',
            ]))
        if has_nvidia and not pt.get('cuda_available'):
            issues.append(('⚠️  NVIDIA GPU detected but CUDA is not responding', [
                'Verify that driver and PyTorch CUDA versions match',
                'Reinstall: https://pytorch.org/get-started/locally/',
            ]))

        if not issues:
            return
        self.print_section('🩺 Smart Diagnostics & Recommendations')
        for title, tips in issues:
            print(f'  {title}')
            for t in tips:
                print(f'     • {t}')
            print()

    # ------------------------------------------------------------------
    # 13–15. Format / model / precision selection
    # ------------------------------------------------------------------

    def get_available_formats(self) -> List[Tuple[str, str, str]]:
        pt = self.env_info['pytorch']
        fw = self.env_info['frameworks']
        formats = []
        if pt.get('installed'):
            if pt.get('cuda_available'):
                formats.append(('pytorch_cuda', 'PyTorch (CUDA)', 'Use .pt file — CUDA acceleration'))
            if pt.get('mps_available'):
                formats.append(('pytorch_mps',  'PyTorch (MPS)',  'Use .pt file — Apple Metal acceleration'))
            if pt.get('xpu_available'):
                formats.append(('pytorch_xpu',  'PyTorch (XPU)',  'Use .pt file — Intel XPU acceleration'))
            formats.append(('pytorch_cpu', 'PyTorch (CPU)', 'Use .pt file — CPU inference'))
        if pt.get('cuda_available') and fw['tensorrt']['available']:
            formats.append(('tensorrt', 'TensorRT', 'Export to .engine — best NVIDIA performance ⭐'))
        if fw['coreml']['available']:
            formats.append(('coreml', 'CoreML', 'Export to .mlpackage — best Apple performance ⭐'))
        if fw['openvino']['available']:
            formats.append(('openvino', 'OpenVINO', 'Export to OpenVINO folder — best Intel/AMD performance ⭐'))
        if pt.get('installed'):
            formats.append(('onnx', 'ONNX', 'Export to .onnx — universal format'))
        return formats

    def select_format(self, preset: str = None) -> Tuple[str, str]:
        formats = self.get_available_formats()
        if not formats:
            print('\n❌ No available formats! Please install PyTorch first.')
            sys.exit(1)

        # Non-interactive: use preset format id if provided
        if preset:
            for fid, name, desc in formats:
                if fid == preset or fid.startswith(preset):
                    print(f'  Format: {name} (from --format)')
                    return fid, name
            print(f'⚠ Requested format "{preset}" is not available; falling back to default.')

        self.print_header('📦 Select Export Format')
        print('Available formats:\n')
        for i, (fid, name, desc) in enumerate(formats, 1):
            print(f'  [{i}] {name}\n      {desc}\n')
        while True:
            try:
                choice = input(f'Select format (1-{len(formats)}) [1]: ').strip()
                choice = int(choice) if choice else 1
                if 1 <= choice <= len(formats):
                    sel = formats[choice - 1]
                    print(f'\n✅ Selected: {sel[1]}')
                    return sel[0], sel[1]
                print(f'❌ Please enter a number between 1 and {len(formats)}')
            except ValueError:
                print('❌ Please enter a valid number')
            except KeyboardInterrupt:
                print('\n\nCancelled'); sys.exit(0)

    def select_model(self,
                     preset_task: str = None,
                     preset_size: str = None,
                     preset_model: str = None) -> str:
        """
        Two-level interactive menu:
          Level 1 — choose the YOLO task (Detection / Pose / Segmentation)
          Level 2 — choose the model size (Nano → XLarge)
        Returns the composed model filename, e.g. 'yolo26s-pose.pt'.
        The user may also type a custom filename at the size prompt to skip
        the automatic composition logic entirely.

        Non-interactive parameters:
          preset_model — direct model filename (skips both levels)
          preset_task  — 'detect' | 'pose' | 'segment'  (skips Level 1)
          preset_size  — 'n' | 's' | 'm' | 'l' | 'x'   (skips Level 2)
        """
        # ── Direct model path ────────────────────────────────────────────────
        if preset_model:
            print(f'  Model: {preset_model} (from --model)')
            return preset_model if preset_model.endswith('.pt') else preset_model + '.pt'

        self.print_header('🎯 Select Task & Model')

        # ── Level 1: task ───────────────────────────────────────────────────
        tasks = {
            '1': ('',      'Detection / Tracking',
                  'Detect and track objects in video  (yolo26?.pt)'),
            '2': ('-pose', 'Pose Estimation',
                  'Detect people and estimate 17-keypoint body pose  (yolo26?-pose.pt)'),
            '3': ('-seg',  'Segmentation',
                  'Detect objects and produce per-pixel instance masks  (yolo26?-seg.pt)'),
        }
        _task_map = {'detect': '1', 'pose': '2', 'segment': '3'}

        if preset_task and preset_task in _task_map:
            t_choice = _task_map[preset_task]
            task_suffix, task_name, _ = tasks[t_choice]
            print(f'  Task: {task_name} (from --task)')
        else:
            print('Step 1 — Select task:\n')
            for k, (_, name, desc) in tasks.items():
                print(f'  [{k}] {name}\n      {desc}\n')
            while True:
                try:
                    t_choice = input('Select task (1-3) [1]: ').strip() or '1'
                    if t_choice in tasks:
                        task_suffix, task_name, _ = tasks[t_choice]
                        print(f'\n✅ Task: {task_name}')
                        break
                    print('❌ Please enter 1, 2, or 3')
                except KeyboardInterrupt:
                    print('\n\nCancelled'); sys.exit(0)

        # ── Level 2: size ───────────────────────────────────────────────────
        sizes = {
            '1': ('n', 'Nano   — fastest, lowest accuracy'),
            '2': ('s', 'Small  — speed / accuracy balance'),
            '3': ('m', 'Medium — better accuracy'),
            '4': ('l', 'Large  — high accuracy'),
            '5': ('x', 'XLarge — highest accuracy'),
        }
        _size_map = {'n': '1', 's': '2', 'm': '3', 'l': '4', 'x': '5'}

        if preset_size and preset_size in _size_map:
            sz = preset_size
            model = f'yolo26{sz}{task_suffix}.pt'
            print(f'  Size: {model} (from --size)')
            return model

        print(f'\nStep 2 — Select model size:\n')
        for k, (sz, desc) in sizes.items():
            name = f'yolo26{sz}{task_suffix}.pt'
            print(f'  [{k}] {name:<22} {desc}')

        print('\n  Or type a custom model filename directly.')
        while True:
            try:
                s_choice = input('\nSelect size (1-5) or filename [2]: ').strip() or '2'
                if s_choice in sizes:
                    sz = sizes[s_choice][0]
                    model = f'yolo26{sz}{task_suffix}.pt'
                    print(f'✅ Selected: {model}')
                    return model
                # Custom filename
                if not s_choice.endswith('.pt'):
                    s_choice += '.pt'
                print(f'✅ Selected: {s_choice}')
                return s_choice
            except KeyboardInterrupt:
                print('\n\nCancelled'); sys.exit(0)

    def select_precision(self, format_id: str, preset: str = None) -> str:
        self.print_header('⚙️  Select Precision')
        configs = {
            'tensorrt': ({'1':'fp32','2':'fp16','3':'int8'}, '2',
                         'TensorRT:\n  [1] FP32  [2] FP16 ⭐  [3] INT8'),
            'onnx':     ({'1':'fp32','2':'fp16'}, '2',
                         'ONNX (INT8 not supported):\n  [1] FP32  [2] FP16 ⭐'),
            'openvino': ({'1':'fp32','2':'fp16','3':'int8'}, '2',
                         'OpenVINO:\n  [1] FP32  [2] FP16 ⭐  [3] INT8'),
            'coreml':   ({'1':'fp32','2':'fp16','3':'int8'}, '2',
                         'CoreML:\n  [1] FP32  [2] FP16 ⭐  [3] INT8'),
        }
        if format_id not in configs:
            return 'fp32'
        opts, dflt, msg = configs[format_id]
        valid_precisions = list(opts.values())

        # Non-interactive: use preset if valid for this format
        if preset:
            if preset in valid_precisions:
                print(f'  Precision: {preset.upper()} (from --precision)')
                return preset
            print(f'⚠ Precision "{preset}" is not valid for {format_id} '
                  f'(valid: {", ".join(valid_precisions)}); using default.')

        print(msg)
        while True:
            try:
                choice = input(f'Select (1-{len(opts)}) [{dflt}]: ').strip()
                if not choice: choice = dflt
                if choice in opts:
                    print(f'✅ Selected: {opts[choice].upper()}')
                    return opts[choice]
                print(f'❌ Please enter a number between 1 and {len(opts)}')
            except KeyboardInterrupt:
                print('\n\nCancelled'); sys.exit(0)

    # ------------------------------------------------------------------
    # 16. Export parameter preparation
    # ------------------------------------------------------------------

    def prepare_export_args(self, format_id: str, model_name: str, precision: str) -> tuple:
        base = model_name.replace('.pt', '')
        if format_id == 'tensorrt':
            args = {'format':'engine','imgsz':EXPORT_IMGSZ,'dynamic':TENSORRT_DYNAMIC,
                    'workspace':TENSORRT_WORKSPACE,'batch':TENSORRT_BATCH}
            if precision == 'fp16': args['half'] = True
            elif precision == 'int8': args.update({'int8':True,'data':CALIBRATION_DATA})
            return args, f'{base}_{precision}.engine'

        elif format_id == 'coreml':
            args = {'format':'coreml','imgsz':EXPORT_IMGSZ,'nms':COREML_NMS,'batch':COREML_BATCH}
            if precision == 'fp16': args['half'] = True
            elif precision == 'int8': args.update({'int8':True,'data':CALIBRATION_DATA})
            return args, f'{base}_{precision}.mlpackage'

        elif format_id == 'openvino':
            args = {'format':'openvino','imgsz':EXPORT_IMGSZ,'dynamic':OPENVINO_DYNAMIC,
                    'nms':OPENVINO_NMS,'batch':OPENVINO_BATCH}
            if precision == 'fp16': args['half'] = True
            elif precision == 'int8':
                args.update({'int8':True,'data':CALIBRATION_DATA,'fraction':OPENVINO_FRACTION})
            return args, f'{base}_{precision}_openvino_model'

        elif format_id == 'onnx':
            args = {'format':'onnx','imgsz':EXPORT_IMGSZ,'dynamic':ONNX_DYNAMIC,
                    'simplify':ONNX_SIMPLIFY,'nms':ONNX_NMS,'batch':ONNX_BATCH}
            if ONNX_OPSET is not None: args['opset'] = ONNX_OPSET
            if precision == 'fp16': args['half'] = True
            return args, f'{base}_{precision}.onnx'

        return {'format':'onnx'}, f'{base}.onnx'

    # ------------------------------------------------------------------
    # 17. Execute export
    # ------------------------------------------------------------------

    def export_model(self, format_id: str, model_name: str, precision: str):
        self.print_header('🚀 Starting Model Export')
        try:
            from ultralytics import YOLO
        except ImportError:
            print('❌ ultralytics not installed  →  pip install ultralytics\n')
            sys.exit(1)
        print(f'Model: {model_name}  Format: {format_id.upper()}  Precision: {precision.upper()}\n')
        try:
            print(f'📥 Loading {model_name}...')
            model = YOLO(model_name)
            args, expected = self.prepare_export_args(format_id, model_name, precision)
            print(f'⚙️  Export parameters: {args}\n⏳ Exporting...\n')
            import time
            t0  = time.time()
            out = model.export(**args)
            elapsed = time.time() - t0
            out_str = str(out)
            if os.path.exists(out_str) and out_str != expected:
                import shutil
                if os.path.isdir(out_str):
                    if os.path.exists(expected): shutil.rmtree(expected)
                    os.rename(out_str, expected)
                else:
                    if os.path.exists(expected): os.remove(expected)
                    os.rename(out_str, expected)
                out = expected
            print(f"\n{'='*70}")
            print(f'✅ Export successful! (elapsed {elapsed:.1f}s)')
            print(f"{'='*70}\n📁 Output: {out}\n")
            self._show_usage(format_id, str(out))
        except Exception as e:
            print(f'\n❌ Export failed: {e}\n')
            import traceback; traceback.print_exc()
            sys.exit(1)

    def _show_usage(self, format_id: str, output_path: str):
        self.print_section('📖 Usage Example')
        print(f"from ultralytics import YOLO\nmodel = YOLO('{output_path}')")
        print("results = model('image.jpg')\nresults[0].show()\n")
        if format_id == 'openvino':
            print("# OpenVINO: device='intel:cpu' / 'intel:gpu' / 'intel:npu'\n")

    def _show_pytorch_usage(self, model_name: str, format_id: str):
        self.print_section('📖 Usage Example')
        dev = {'pytorch_cuda':'cuda','pytorch_mps':'mps',
               'pytorch_xpu':'xpu','pytorch_cpu':'cpu'}.get(format_id,'cpu')
        print(f"from ultralytics import YOLO\nmodel = YOLO('{model_name}')")
        print(f"results = model('image.jpg', device='{dev}')\nresults[0].show()\n")

    # ------------------------------------------------------------------
    # 18. Main flow
    # ------------------------------------------------------------------

    def run(self, cli_args=None):
        """
        Main flow.  When *cli_args* (an argparse.Namespace) is supplied,
        all interactive prompts are bypassed for arguments that were provided
        on the command line.  Pass ``--auto`` to skip every prompt silently.
        """
        print('\n' + '='*70)
        print('  🔥 YOLO Environment Checker & Model Export Tool  v1.1.0')
        print('='*70)
        self.detect_all()
        self.show_environment()

        # ── Scan-only mode: show environment then exit ──────────────────────
        if cli_args and cli_args.scan_only:
            return

        if not self.env_info['pytorch'].get('installed'):
            print('\n❌ PyTorch is not installed; cannot proceed to export.')
            print('   pip install torch\n')
            sys.exit(1)

        # ── Determine whether to skip the "Press Enter" gate ────────────────
        non_interactive = cli_args and (
            cli_args.auto or cli_args.format or cli_args.task
            or cli_args.size or cli_args.model
        )

        if not non_interactive:
            self.print_section('📝 Next Step')
            print('You will now select a model and export format.\n')
            try:
                input('Press Enter to continue, or Ctrl+C to cancel... ')
            except KeyboardInterrupt:
                print('\n\nCancelled'); sys.exit(0)

        # ── Format selection ─────────────────────────────────────────────────
        preset_fmt = None
        if cli_args:
            if cli_args.format:
                preset_fmt = cli_args.format
            elif cli_args.auto:
                # Pick the first (highest-priority) available format automatically
                avail = self.get_available_formats()
                preset_fmt = avail[0][0] if avail else None

        format_id, format_name = self.select_format(preset=preset_fmt)

        # ── Model selection ──────────────────────────────────────────────────
        preset_task  = cli_args.task  if cli_args else None
        preset_size  = cli_args.size  if cli_args else None
        preset_model = cli_args.model if cli_args else None

        model_name = self.select_model(
            preset_task=preset_task,
            preset_size=preset_size,
            preset_model=preset_model,
        )

        if format_id.startswith('pytorch_'):
            self.print_header('🚀 Using PyTorch Model Directly')
            dev_map = {'pytorch_cuda':'CUDA','pytorch_mps':'MPS',
                       'pytorch_xpu':'XPU','pytorch_cpu':'CPU'}
            print(f'Model: {model_name}  Device: {dev_map.get(format_id,format_id)}\n')
            self._show_pytorch_usage(model_name, format_id)
            self.print_header('✨ Done')
            print('Thank you for using the tool!\n')
            return

        # ── Precision selection ──────────────────────────────────────────────
        preset_prec = None
        if cli_args:
            if cli_args.precision:
                preset_prec = cli_args.precision
            elif cli_args.auto:
                preset_prec = 'fp16'   # sensible default for all export formats

        precision = self.select_precision(format_id, preset=preset_prec)
        self.export_model(format_id, model_name, precision)
        self.print_header('✨ Done')
        print('Thank you for using the tool!\n')


# ────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='yolo_env_checker',
        description=(
            'YOLO Environment Checker & Model Export Tool  v1.1.0\n\n'
            'Without arguments the tool runs interactively.\n'
            'Supply --auto (or individual flags) for non-interactive / CI use.\n\n'
            'Examples:\n'
            '  # Interactive (default)\n'
            '  python yolo_env_checker.py\n\n'
            '  # Scan only — print env report and exit\n'
            '  python yolo_env_checker.py --scan-only\n\n'
            '  # Fully automated export\n'
            '  python yolo_env_checker.py --auto\n\n'
            '  # Specify every option explicitly\n'
            '  python yolo_env_checker.py --task detect --size s '
            '--format tensorrt --precision fp16\n\n'
            '  # Export a pose model to ONNX/FP32 without prompts\n'
            '  python yolo_env_checker.py --task pose --size m '
            '--format onnx --precision fp32'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        '--scan-only', action='store_true',
        help='Print environment report and exit without exporting.',
    )
    p.add_argument(
        '--auto', action='store_true',
        help=(
            'Skip all interactive prompts and use defaults / best available '
            'values.  Individual flags (--task, --size, etc.) override the '
            'automatic choices.'
        ),
    )
    p.add_argument(
        '--task', choices=['detect', 'pose', 'segment'], default=None,
        metavar='TASK',
        help='YOLO task: detect | pose | segment  (default: detect)',
    )
    p.add_argument(
        '--size', choices=['n', 's', 'm', 'l', 'x'], default=None,
        metavar='SIZE',
        help='Model size: n | s | m | l | x  (default: s)',
    )
    p.add_argument(
        '--model', default=None, metavar='FILENAME',
        help=(
            'Direct model filename (e.g. yolo26s-pose.pt).  '
            'When set, --task and --size are ignored.'
        ),
    )
    p.add_argument(
        '--format',
        choices=['pytorch', 'pytorch_cuda', 'pytorch_mps', 'pytorch_xpu',
                 'pytorch_cpu', 'tensorrt', 'coreml', 'openvino', 'onnx'],
        default=None, metavar='FORMAT',
        help=(
            'Export format: pytorch | tensorrt | coreml | openvino | onnx  '
            '(default: best available)'
        ),
    )
    p.add_argument(
        '--precision', choices=['fp32', 'fp16', 'int8'], default=None,
        metavar='PREC',
        help='Export precision: fp32 | fp16 | int8  (default: fp16)',
    )
    return p


def main():
    cli_args = _build_arg_parser().parse_args()
    try:
        YOLOEnvChecker().run(cli_args=cli_args)
    except KeyboardInterrupt:
        print('\n\nCancelled'); sys.exit(0)
    except Exception as e:
        print(f'\n❌ Fatal error: {e}\n')
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
