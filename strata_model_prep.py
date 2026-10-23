#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
strata_model_prep.py
======================
Automation tool for preparing models downloaded from Hugging Face and
generating JSON configuration files for the Strata local inference engine.

Compatibility: Python 3.8+ on Linux and Windows (basic support for macOS).
Official Integration: runs Strata's own tools (tools/iq_pack.py, tools/mtp_*.py,
tools/gguf_reader.py) and reuses setup.py's rules (steps 6 and 7) where they can
be imported, so the configs it writes are the ones 'serve/server.py' and the
engine accept.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import readline
    HAVE_READLINE = True
except ImportError:
    HAVE_READLINE = False


# ==============================================================================
# 1. STRATA CONSTANTS AND DEFAULT PATHS
# ==============================================================================

WIN = os.name == "nt"
STRATA_CONFIG_FILENAME = "settings.json"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STRATA_ROOT = Path("/home/USERNAME/Documents/AI/bin/Strata")
DEFAULT_DATA_DIR = Path("/home/USERNAME/Documents/AI/llm_models/Strata-data")

# Values the engine and setup.py use (src/program/generate.cpp, setup.py)
ENGINE_VRAM_RESERVE_MIB = 700          # generate.cpp: vram_reserve_mib default (Linux)
VISION_GPU_RESERVE_MIB = 700           # setup.py VISION["gpu"]["reserve_mib"]
VISION_MAX_TOKENS = {"gpu": 1024, "cpu": 300}
KV_RESIDENT_TOKENS = 32768             # setup.py step 7: KV streaming writes --kv-resident 32768
ENGINE_KV_RESIDENT_MIN = 20480         # src/core/layer.cpp qsa_kv_resident_min()
KV_STREAM_MIN_CTX = 65536              # setup.py: KV streaming from 64K up
KV_CELL_BYTES = {"fp16": 2112, "int8": 1056, "q4_0": 576, "k8v4": 816}   # setup.py UD_FIT_KV_BYTES
KV_ATTN_LAYERS = 13                    # 12 QSA layers + the MTP draft layer (setup.py kv_streaming_ram_gb)
DEFAULT_KV_QUANT = "int8"
DEFAULT_PREFILL_CHUNK = "auto"
DEFAULT_SPEC_MIN_P = "0.5"
DEFAULT_SPEC_DRAFTS = "4"
DEFAULT_REASONING_BUDGET = 8192
MODEL_RAM_BESIDE_EXPERTS_GB = 12       # setup.py MODELS: ram_gb - arena_gb (IQ3_S 62 - 50.3, IQ2_XS 48 - 35.5)
UNSLOTH_RESIDENT_BESIDE_GB = 16        # setup.py: every Unsloth expert in RAM when they fit with 16 GB beside them
UNSLOTH_RAM_LEFT_GB = 24               # setup.py resident_budget_gib

SUPPORTED_KV_QUANTS = ["int8", "q4_0", "k8v4", "fp16"]   # generate.cpp: --kv must be fp16, int8, q4_0 or k8v4

# Constants aligned with Strata's setup.py (Source of Truth)
TRAINED_CONTEXT_LIMIT = 262144
STANDARD_CONTEXTS = [8192, 32768, 65536, 131072, 204800, 262144, 393216, 524288]   # setup.py CONTEXTS
DEFAULT_CONTEXT = 131072
DRAFT_VOCABS = {
    "cjk": "draft_vocab.bin",
    "en": "draft_vocab_en.bin",
    "cyrillic": "draft_vocab_cyrillic.bin",
    "fr": "draft_vocab_fr.bin",
}
OLD_DRAFT_VOCABS = {"369151522226a5edaa5f12cfd1e2ae7db8f4fbdbd222f3dcf327dced9597fb25"}   # setup.py: to 0.1.26

PLE_TENSOR = "per_layer_token_embd.weight"
SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE)
PACK_SOURCE_FILE = "model-config-source.json"   # written by this script: which GGUF a pack was built from


class TermColor:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"


def color_text(text: str, color: str) -> str:
    if sys.stdout.isatty():
        return f"{color}{text}{TermColor.RESET}"
    return text


def safe_input(prompt: str, default: str = "") -> str:
    """Prompt user input safely against EOFError and cancellation."""
    try:
        val = input(prompt).strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        print()
        return default


def _path_completer(text: str, state: int) -> Optional[str]:
    """Readline tab-completer for filesystem paths (files and directories)."""
    expanded = os.path.expanduser(text)
    if not text:
        try:
            entries = os.listdir(".")
            candidates = [f + ("/" if os.path.isdir(f) else "") for f in entries if not f.startswith(".")]
        except Exception:
            return None
    else:
        dirname, rest = os.path.split(expanded)
        search_dir = dirname if dirname else "."
        try:
            entries = os.listdir(search_dir)
        except Exception:
            return None

        orig_dirname, _ = os.path.split(text)
        candidates = []
        for entry in entries:
            if not rest and entry.startswith(".") and not text.endswith("/."):
                continue
            if entry.startswith(rest):
                full_path = os.path.join(search_dir, entry)
                disp_path = os.path.join(orig_dirname, entry) if orig_dirname else entry
                if os.path.isdir(full_path) and not disp_path.endswith("/"):
                    disp_path += "/"
                candidates.append(disp_path)

    candidates.sort()
    try:
        return candidates[state]
    except IndexError:
        return None


def safe_input_path(prompt: str, default: str = "") -> str:
    """Prompt user input with TAB autocompletion enabled for filesystem paths."""
    if not sys.stdin.isatty() or not HAVE_READLINE:
        return safe_input(prompt, default)

    old_completer = readline.get_completer()
    old_delims = readline.get_completer_delims()
    try:
        readline.set_completer_delims(" \t\n;\"'")
        if readline.__doc__ and "libedit" in readline.__doc__:
            readline.parse_and_bind("bind ^I rl_complete")
        else:
            readline.parse_and_bind("tab: complete")
        readline.set_completer(_path_completer)
        return safe_input(prompt, default)
    finally:
        readline.set_completer(old_completer)
        readline.set_completer_delims(old_delims)


_SETUP_MODULE: Dict[str, Any] = {}


def load_setup(strata_root: Optional[Path]) -> Any:
    """Strata's own setup.py as a module (its rules for pool workers, calibration, rope and config carry-over), or
    None when it cannot be imported (then this script's copies of those rules are used)."""
    if not strata_root:
        return None
    key = str(strata_root)
    if key in _SETUP_MODULE:
        return _SETUP_MODULE[key]
    mod = None
    path = strata_root / "setup.py"
    if path.is_file():
        try:
            spec = importlib.util.spec_from_file_location("strata_setup", str(path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            print(color_text(f"[i] setup.py could not be imported ({e}); using this script's copies of its rules.",
                             TermColor.YELLOW))
            mod = None
    _SETUP_MODULE[key] = mod
    return mod


def determine_model_config_filename(
    model_or_pack_path: Union[Path, str],
    native_shard: Optional[Union[Path, str]] = None,
    pack_path: Optional[Union[Path, str]] = None,
) -> str:
    """
    Determines the JSON configuration filename matching the model filename.
    Example: 'Qwen3.8-Flash-Next-Uncensored.i1-IQ3_M.gguf' -> 'Qwen3.8-Flash-Next-Uncensored.i1-IQ3_M.json'
    """
    model_or_pack_path = Path(model_or_pack_path)
    native_shard = Path(native_shard) if native_shard else None
    pack_path = Path(pack_path) if pack_path else None

    candidate_file = None
    if model_or_pack_path.name.lower().endswith(".gguf"):
        candidate_file = model_or_pack_path
    elif native_shard and native_shard.name.lower().endswith(".gguf"):
        candidate_file = native_shard

    if candidate_file:
        return f"{gguf_base_name(candidate_file)}.json"

    # Fallback to pack or folder name
    fallback_dir = pack_path or model_or_pack_path
    clean_dir = fallback_dir.name
    if clean_dir.lower().endswith(".gguf"):
        clean_dir = clean_dir[:-5]
    clean_dir = re.sub(r"-\d{5}-of-\d{5}$", "", clean_dir, flags=re.IGNORECASE)
    return f"{clean_dir}.json"


# ==============================================================================
# 2. HARDWARE DETECTION
# ==============================================================================

class HardwareInfo:
    def __init__(self):
        self.os_name: str = platform.system()
        self.total_ram_bytes: int = 0
        self.available_ram_bytes: int = 0
        self.safe_ram_bytes: int = 0
        self.gpus: List[Dict[str, Any]] = []
        self.cpu_name: str = ""
        self.cpu_cores: int = os.cpu_count() or 1

    @property
    def total_ram_gb(self) -> float:
        return self.total_ram_bytes / (1024 ** 3)

    @property
    def available_ram_gb(self) -> float:
        return self.available_ram_bytes / (1024 ** 3)

    @property
    def safe_ram_gb(self) -> float:
        return self.safe_ram_bytes / (1024 ** 3)

    @property
    def primary_gpu(self) -> Optional[Dict[str, Any]]:
        """The card Strata runs on by default: the one with the most VRAM, ties to the lower number (setup.py
        gpu_info)."""
        if not self.gpus:
            return None
        return sorted(self.gpus, key=lambda g: (-g["vram_total_mib"], str(g["index"])))[0]


class HardwareDetector:
    """Detects system memory, CPU architecture, and available GPU accelerators."""

    @staticmethod
    def detect() -> HardwareInfo:
        hw = HardwareInfo()
        HardwareDetector._detect_ram(hw)
        HardwareDetector._detect_cpu(hw)
        HardwareDetector._detect_gpus(hw)
        return hw

    @staticmethod
    def _detect_ram(hw: HardwareInfo) -> None:
        if hw.os_name == "Linux" and os.path.exists("/proc/meminfo"):
            try:
                meminfo = {}
                with open("/proc/meminfo", "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.split(":")
                        if len(parts) == 2:
                            key = parts[0].strip()
                            val = parts[1].strip().split()[0]
                            meminfo[key] = int(val) * 1024
                hw.total_ram_bytes = meminfo.get("MemTotal", 0)
                hw.available_ram_bytes = meminfo.get("MemAvailable", hw.total_ram_bytes // 2)
            except Exception:
                pass

        if hw.total_ram_bytes == 0:
            try:
                import psutil
                vm = psutil.virtual_memory()
                hw.total_ram_bytes = vm.total
                hw.available_ram_bytes = vm.available
            except ImportError:
                hw.total_ram_bytes = 16 * (1024 ** 3)
                hw.available_ram_bytes = 8 * (1024 ** 3)

        os_reserve = max(4 * (1024 ** 3), int(hw.total_ram_bytes * 0.12))
        hw.safe_ram_bytes = max(0, hw.available_ram_bytes - os_reserve)

    @staticmethod
    def _detect_cpu(hw: HardwareInfo) -> None:
        if hw.os_name == "Linux" and os.path.exists("/proc/cpuinfo"):
            try:
                with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if "model name" in line:
                            hw.cpu_name = line.split(":")[1].strip()
                            break
            except Exception:
                pass
        if not hw.cpu_name:
            hw.cpu_name = platform.processor() or "Unknown CPU"

    @staticmethod
    def _detect_gpus(hw: HardwareInfo) -> None:
        if shutil.which("nvidia-smi"):
            try:
                res = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,memory.total,memory.free,driver_version",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                for line in res.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        hw.gpus.append({
                            "type": "NVIDIA",
                            "index": int(parts[0]),
                            "name": parts[1],
                            "vram_total_mib": float(parts[2]),
                            "vram_free_mib": float(parts[3]),
                            "driver": parts[4],
                        })
            except Exception:
                pass

        if not hw.gpus and shutil.which("rocm-smi"):
            try:
                res = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram", "--json"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                data = json.loads(res.stdout)
                for card_id, info in data.items():
                    total_bytes = int(info.get("VRAM Total Memory (B)", 0))
                    used_bytes = int(info.get("VRAM Total Used Memory (B)", 0))
                    m = re.search(r"\d+", str(card_id))
                    hw.gpus.append({
                        "type": "AMD",
                        "index": int(m.group(0)) if m else 0,
                        "name": f"AMD Radeon ({card_id})",
                        "vram_total_mib": total_bytes / (1024 * 1024),
                        "vram_free_mib": (total_bytes - used_bytes) / (1024 * 1024),
                        "driver": "ROCm",
                    })
            except Exception:
                pass


# ==============================================================================
# 3. NATIVE GGUF READER AND MODEL ANALYSIS
# ==============================================================================

class GGUFReader:
    """Lightweight binary reader to extract metadata from GGUF files (v2 & v3) with zero dependencies."""

    GGUF_MAGIC = 0x46554747  # b'GGUF'

    TYPE_UINT8 = 0
    TYPE_INT8 = 1
    TYPE_UINT16 = 2
    TYPE_INT16 = 3
    TYPE_UINT32 = 4
    TYPE_INT32 = 5
    TYPE_FLOAT32 = 6
    TYPE_BOOL = 7
    TYPE_STRING = 8
    TYPE_ARRAY = 9
    TYPE_UINT64 = 10
    TYPE_INT64 = 11
    TYPE_FLOAT64 = 12

    @classmethod
    def read_metadata(cls, file_path: Path, max_keys: int = 250) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        with open(file_path, "rb") as f:
            magic_bytes = f.read(4)
            if len(magic_bytes) < 4:
                return {}
            magic = struct.unpack("<I", magic_bytes)[0]
            if magic != cls.GGUF_MAGIC:
                return {}

            version = struct.unpack("<I", f.read(4))[0]
            if version not in (2, 3):
                return {}

            tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]
            metadata["_version"] = version
            metadata["_tensor_count"] = tensor_count
            metadata["_kv_count"] = kv_count

            for _ in range(min(kv_count, max_keys)):
                key = cls._read_string(f)
                if key is None:
                    break
                vtype = struct.unpack("<I", f.read(4))[0]
                val = cls._read_value(f, vtype)
                metadata[key] = val

        return metadata

    @classmethod
    def _read_string(cls, f) -> Optional[str]:
        length_bytes = f.read(8)
        if len(length_bytes) < 8:
            return None
        length = struct.unpack("<Q", length_bytes)[0]
        str_bytes = f.read(length)
        try:
            return str_bytes.decode("utf-8", errors="replace")
        except Exception:
            return None

    @classmethod
    def _read_value(cls, f, vtype: int) -> Any:
        if vtype == cls.TYPE_UINT8:
            return struct.unpack("<B", f.read(1))[0]
        elif vtype == cls.TYPE_INT8:
            return struct.unpack("<b", f.read(1))[0]
        elif vtype == cls.TYPE_UINT16:
            return struct.unpack("<H", f.read(2))[0]
        elif vtype == cls.TYPE_INT16:
            return struct.unpack("<h", f.read(2))[0]
        elif vtype == cls.TYPE_UINT32:
            return struct.unpack("<I", f.read(4))[0]
        elif vtype == cls.TYPE_INT32:
            return struct.unpack("<i", f.read(4))[0]
        elif vtype == cls.TYPE_FLOAT32:
            return struct.unpack("<f", f.read(4))[0]
        elif vtype == cls.TYPE_BOOL:
            return struct.unpack("<?", f.read(1))[0]
        elif vtype == cls.TYPE_STRING:
            return cls._read_string(f)
        elif vtype == cls.TYPE_UINT64:
            return struct.unpack("<Q", f.read(8))[0]
        elif vtype == cls.TYPE_INT64:
            return struct.unpack("<q", f.read(8))[0]
        elif vtype == cls.TYPE_FLOAT64:
            return struct.unpack("<d", f.read(8))[0]
        elif vtype == cls.TYPE_ARRAY:
            sub_type = struct.unpack("<I", f.read(4))[0]
            arr_len = struct.unpack("<Q", f.read(8))[0]
            if arr_len > 100:
                type_sizes = {
                    cls.TYPE_UINT8: 1, cls.TYPE_INT8: 1,
                    cls.TYPE_UINT16: 2, cls.TYPE_INT16: 2,
                    cls.TYPE_UINT32: 4, cls.TYPE_INT32: 4,
                    cls.TYPE_FLOAT32: 4, cls.TYPE_BOOL: 1,
                    cls.TYPE_UINT64: 8, cls.TYPE_INT64: 8,
                    cls.TYPE_FLOAT64: 8,
                }
                if sub_type in type_sizes:
                    f.seek(type_sizes[sub_type] * arr_len, os.SEEK_CUR)
                    return f"<Array of {arr_len} items>"
            items = []
            for _ in range(arr_len):
                items.append(cls._read_value(f, sub_type))
            return items
        return None


class ModelInfo:
    """Consolidated technical information of the inspected model."""

    def __init__(self, source_path: Path):
        self.source_path = source_path
        self.is_valid: bool = False
        self.format: str = "UNKNOWN"
        self.family: str = "qwen"          # setup.py FAMILIES: qwen, swift, coder, unsloth, nvfp4
        self.architecture: str = "Unknown"
        self.model_name: str = ""
        self.quantization: str = "Unknown"
        self.total_size_bytes: int = 0
        self.ple_bytes: int = 0
        self.num_layers: int = 0
        self.hidden_size: int = 0
        self.num_heads: int = 0
        self.num_kv_heads: int = 0
        self.max_context: int = TRAINED_CONTEXT_LIMIT
        self.is_moe: bool = False
        self.is_coder: bool = False
        self.num_experts: int = 0
        self.active_experts_per_tok: int = 0
        self.gguf_shards: List[Path] = []
        self.mmproj_file: Optional[Path] = None
        self.native_shard: Optional[Path] = None
        self.ple_shard: Optional[Path] = None
        self.pack_dir: Optional[Path] = None
        self.ready: Dict[str, Path] = {}   # ready-made pack (NVFP4): pack, dense, ple, embd
        self.candidates: Dict[str, List[Path]] = {}   # a folder holding several models: base name -> shards
        self.available_packs: List[Path] = []
        self.available_models: List[Path] = []
        self.notes: List[str] = []

    @property
    def total_size_gb(self) -> float:
        return self.total_size_bytes / (1024 ** 3)

    @property
    def experts_gb(self) -> float:
        """What the engine keeps in RAM, in GiB (setup.py's "GB": ram_gb() is MemTotal / 2**30): the model less the
        PLE table, which is read from the disk (setup.py MODELS arena_gb: IQ3_S 50.3 of a 79 GB download)."""
        return max(0, self.total_size_bytes - self.ple_bytes) / 1024 ** 3

    @property
    def head_dim(self) -> int:
        if self.num_heads > 0 and self.hidden_size > 0:
            return self.hidden_size // self.num_heads
        return 128


def extract_quant_tag(name: str) -> str:
    """Extracts normalized quantization tag (e.g. 'iq3_s', 'iq3_xxs', 'iq4_xs', 'q4_k_xl')."""
    n = name.lower().replace("-", "_")
    for q in ["iq1_m", "iq1_s", "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_xs", "iq3_s", "iq3_m", "iq4_xs",
              "iq4_nl", "q4_k_xl", "q4_k_m", "q4_0", "q8_0", "q2_0", "nvfp4", "bf16", "fp16"]:
        if q in n:
            return q
    return ""


def is_aux_gguf(path: Path) -> bool:
    """A GGUF beside the model that is not one of its shards: the vision encoder or an importance matrix."""
    n = path.name.lower()
    return "mmproj" in n or "imatrix" in n


def gguf_base_name(path: Path) -> str:
    """The model's name without '.gguf' and without the '-0000N-of-0000M' shard suffix."""
    m = SHARD_RE.match(path.name)
    if m:
        return m.group("stem")
    return path.name[:-5] if path.name.lower().endswith(".gguf") else path.name


def expand_shards(path: Path) -> Tuple[List[Path], List[str]]:
    """Every shard of the model `path` belongs to, in order, and the names of the ones that are missing. The
    siblings are the exact names <stem>-0000i-of-0000N.gguf (the rule iq_pack.py and the engine use); nothing is
    matched by substring, so '-00001-of-00002' can never be taken for shard 2 or for another model's shard."""
    m = SHARD_RE.match(path.name)
    if not m:
        return [path], ([] if path.exists() else [path.name])
    stem, total = m.group("stem"), int(m.group("total"))
    width = len(m.group("total"))
    shards, missing = [], []
    for i in range(1, total + 1):
        sib = path.with_name(f"{stem}-{i:0{width}d}-of-{total:0{width}d}.gguf")
        if sib.exists():
            shards.append(sib)
        else:
            missing.append(sib.name)
    return shards, missing


def group_gguf_models(dir_path: Path) -> Dict[str, List[Path]]:
    """The models of a folder: base name -> its shards (a folder can hold several quants of one model)."""
    groups: Dict[str, List[Path]] = {}
    for f in sorted(dir_path.glob("*.gguf")):
        if is_aux_gguf(f) or not f.is_file():
            continue
        groups.setdefault(gguf_base_name(f), [])
    for base in list(groups):
        first = next(f for f in sorted(dir_path.glob("*.gguf")) if gguf_base_name(f) == base and not is_aux_gguf(f))
        shards, _ = expand_shards(first)
        groups[base] = shards
    return groups


_TENSOR_CACHE: Dict[str, Optional[Dict[str, int]]] = {}


def gguf_tensors(path: Path, strata_root: Optional[Path]) -> Optional[Dict[str, int]]:
    """{tensor name: bytes} of a GGUF through Strata's tools/gguf_reader.py (the reader setup.py step 7 uses), or
    None when the reader is not available or cannot read the file."""
    key = str(path)
    if key in _TENSOR_CACHE:
        return _TENSOR_CACHE[key]
    out = None
    root = strata_root or StrataEnvironment.find_strata_root()
    if root and (root / "tools" / "gguf_reader.py").exists():
        tools_path = str(root / "tools")
        if tools_path not in sys.path:
            sys.path.insert(0, tools_path)
        try:
            from gguf_reader import GGUFFile
            gf = GGUFFile(path)
            out = {}
            for t in gf.tensors:
                try:
                    out[t.name] = int(t.expected_bytes())
                except Exception:
                    out[t.name] = 0
        except Exception:
            out = None
    _TENSOR_CACHE[key] = out
    return out


def find_ple_shard(shards: List[Path], strata_root: Optional[Path] = None) -> Optional[Path]:
    """The shard holding 'per_layer_token_embd.weight' (--ple-gguf), as setup.py step 7 finds it: by the tensor
    itself.  The original Qwen / Coder / Unsloth files keep it in shard 2, Swift 1.5 in shard 1 (iq_pack.py's
    header); the file names are only a fallback when tools/gguf_reader.py cannot be used.  None: no shard has it."""
    if not shards:
        return None
    reader_ok = True
    for s in shards:
        tensors = gguf_tensors(s, strata_root)
        if tensors is None:
            reader_ok = False
            break
        if PLE_TENSOR in tensors:
            return s
    if reader_ok:
        return None
    if len(shards) == 1:
        return shards[0]
    by_index = {int(m.group("idx")): s for s in shards for m in [SHARD_RE.match(s.name)] if m}
    if "swift" in shards[0].name.lower():
        return by_index.get(1, shards[0])
    return by_index.get(2, shards[1])


def read_pack_header(pack_dir: Path) -> Tuple[Optional[str], List[str]]:
    """(the shard-1 file name a pack was built from, the other shard names its native_experts.txt names), from the
    header iq_pack.py writes: '... absolute offsets in <name>, or in the named shard beside it ...'.  Rows are
    'layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard | gate,up,down]'."""
    native_txt = pack_dir / "native_experts.txt"
    src, others = None, []
    try:
        with open(native_txt, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline()
            m = re.search(r"absolute offsets in (.+?\.gguf)", first, re.IGNORECASE)
            if m:
                src = m.group(1).strip()
            for line in f:
                parts = line.split()
                if len(parts) >= 9:
                    for name in parts[8].split(","):
                        if name.lower().endswith(".gguf") and name not in others and name != src:
                            others.append(name)
    except OSError:
        pass
    return src, others


def read_pack_source(pack_dir: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads((pack_dir / PACK_SOURCE_FILE).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def human_size(n: int) -> str:
    """1,466.78 MiB / 3.2 KiB / 330 B: a size in the largest unit that keeps it at 1 or more."""
    for unit, scale in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= scale:
            return f"{n / scale:,.2f} {unit}"
    return f"{n} B"


def done_mark(path: Path) -> Path:
    """A file's finish mark, as setup.py writes it (done() / mark()): <file>.done beside it."""
    return path.with_name(path.name + ".done")


def gguf_problem(path: Path, strata_root: Optional[Path]) -> Optional[str]:
    """setup.py check_shards: why a GGUF shard is not whole, or None.  Whole means as long as its own tensor directory
    says (only the header is read); a shard of one tensor (the PLE table) is exactly that long (#657)."""
    root = strata_root or StrataEnvironment.find_strata_root()
    if not path.exists():
        return f"missing {path}"
    if not root or not (root / "tools" / "gguf_reader.py").exists():
        return None
    tools_path = str(root / "tools")
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    try:
        from gguf_reader import GGUFFile
        g = GGUFFile(path)
    except (ImportError, ValueError, OSError, struct.error) as e:
        return f"{path.name} is not a whole GGUF shard ({e})"
    need = g.data_start + max((t.offset + (t.expected_bytes() or 0) for t in g.tensors), default=0)
    have = path.stat().st_size
    if not g.tensors:          # #2346: a metadata-only shard ends at its header
        need = min(need, have)
    if have < need:
        return f"{path.name} is short: {have:,} of {need:,} bytes ({need - have:,} missing)"
    if len(g.tensors) == 1 and have - need >= 64:
        return f"{path.name} is longer than its tensor: {have:,} bytes, {need:,} expected ({have - need:,} extra)"
    return None


def mark_done(paths: List[Path]) -> Tuple[List[Path], List[Path]]:
    """setup.py mark(): <file>.done holding the date (YYYY-MM-DD HH:MM), for each file this script processed.  A mark
    already there is kept (setup.py's own, or one holding a verified SHA-256); a folder that cannot be written only
    costs the mark.  (marks written, marks already there)."""
    written, kept = [], []
    for p in paths:
        m = done_mark(p)
        if m.exists():
            kept.append(m)
            continue
        try:
            m.write_text(time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
            written.append(m)
        except OSError as e:
            print(color_text(f"  [!] the finish mark of {p.name} cannot be written ({e})", TermColor.YELLOW))
    return written, kept


def pack_complete(pack_dir: Path) -> bool:
    """iq_pack.py writes native_experts.txt last (its completion mark) and the tokenizer from the GGUF."""
    return all((pack_dir / f).exists() for f in ("index.txt", "dense.bin", "native_experts.txt")) and \
        (pack_dir / "tokenizer" / "vocab.json").exists()


def same_file(a: Union[Path, str], b: Union[Path, str]) -> bool:
    """One file, whichever folder link it is reached through (bin/Strata-data and llm_models/Strata-data here)."""
    try:
        return os.path.samefile(str(a), str(b))
    except OSError:
        return os.path.abspath(str(a)) == os.path.abspath(str(b))


STRONG_MATCHES = ("source record", "conversions.json shards")


def pack_matches(pack_dir: Path, native_shard: Path) -> Tuple[bool, str]:
    """Whether a pack was built from this GGUF, from the strongest evidence the pack has:
    1. this script's source record (the shard's absolute path);
    2. iq_pack.py's conversions.json "source_shards": every shard's name and size in bytes (setup.py's packs have it);
    3. the shard name in native_experts.txt alone - weak: two models can share a file name (Swift and Huihui's Swift,
       Qwen and Coder)."""
    src = read_pack_source(pack_dir)
    if src and src.get("native"):
        same = same_file(src["native"], native_shard)
        return same, ("source record" if same else f"built from {src['native']}")
    try:
        conv = json.loads((pack_dir / "conversions.json").read_text(encoding="utf-8"))
        recorded = [(s["name"], int(s["size"])) for s in conv.get("source_shards") or []]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        recorded = []
    if recorded:
        shards, _ = expand_shards(native_shard)
        actual = [(s.name, s.stat().st_size) for s in shards if s.exists()]
        if recorded == actual:
            return True, "conversions.json shards"
        return False, "built from " + ", ".join(f"{n} ({sz:,} B)" for n, sz in recorded)
    name, _ = read_pack_header(pack_dir)
    if name and name == native_shard.name:
        return True, "file name only (no source record)"
    return False, f"built from {name or 'an unknown file'}"


def detect_family(info: ModelInfo) -> str:
    """setup.py FAMILIES for a GGUF: nvfp4 (ready pack), swift, coder (256 experts), unsloth (UD- quants), qwen."""
    if info.ready:
        return "nvfp4"
    name = info.native_shard.name if info.native_shard else info.source_path.name
    if "swift" in name.lower():
        return "swift"
    if info.num_experts == 256 or ("coder" in str(info.source_path).lower() and not info.num_experts):
        return "coder"
    if re.search(r"(^|[-_.])UD-", name) or ("unsloth" in str(info.source_path).lower() and
                                           len(info.gguf_shards) > 2):
        return "unsloth"
    return "qwen"


class ModelAnalyzer:
    """Inspects local directories downloaded from Hugging Face or Strata pack artifacts."""

    @classmethod
    def inspect(cls, target_path: Path, strata_root: Optional[Path] = None,
                data_dir: Optional[Path] = None) -> ModelInfo:
        target_path = Path(target_path).expanduser().absolute()
        info = ModelInfo(target_path)

        if not target_path.exists():
            info.notes.append(f"Path does not exist: {target_path}")
            return info

        if target_path.is_file() and target_path.suffix.lower() == ".gguf":
            if is_aux_gguf(target_path):
                info.notes.append(f"{target_path.name} is a vision encoder or importance matrix, not the model.")
                return info
            shards, missing = expand_shards(target_path)
            if missing:
                info.notes.append(f"Missing shard(s) beside {target_path.name}: {', '.join(missing)}")
                return info
            cls._inspect_gguf_set(shards, info, strata_root, data_dir)
            return info

        if target_path.is_file() and target_path.name == "native_experts.txt":
            target_path = target_path.parent
            info.source_path = target_path

        if target_path.is_dir():
            # 1. A ready-made pack (NVFP4): pack/ beside the dense, PLE and embedding GGUFs
            ready = cls._ready_paths(target_path)
            if ready:
                cls._inspect_ready(ready, info, strata_root, data_dir)
                return info

            # 2. Check if it is the root data directory (Strata-data)
            has_sub_models = (target_path / "models").is_dir()
            has_sub_packs = (target_path / "packs").is_dir() or (target_path / "artifacts").is_dir()
            if has_sub_models or has_sub_packs:
                info.format = "STRATA_DATA_ROOT"
                info.is_valid = False
                packs_dir = target_path / "packs"
                models_dir = target_path / "models"
                for pd in (target_path / "artifacts", packs_dir):
                    if pd.is_dir():
                        info.available_packs += sorted(p for p in [pd, *pd.iterdir()]
                                                       if p.is_dir() and (p / "index.txt").exists())
                if models_dir.is_dir():
                    info.available_models = sorted([m for m in models_dir.iterdir() if m.is_dir() and any(m.glob("*.gguf"))])
                info.notes.append(f"The path '{target_path}' is the root directory of Strata-data, not an individual pack or model.")
                if info.available_packs:
                    info.notes.append(f"Available packs: {', '.join(p.name for p in info.available_packs)}")
                if info.available_models:
                    info.notes.append(f"Available models: {', '.join(m.name for m in info.available_models)}")
                return info

            # 3. Check if it is a valid Strata Pack
            if (target_path / "index.txt").exists() and (target_path / "dense.bin").exists():
                cls._inspect_strata_pack(target_path, info, strata_root, data_dir)
                return info

            # 4. Check if it is a directory of GGUF model shards (one model, or several to choose from)
            groups = group_gguf_models(target_path)
            if len(groups) == 1:
                shards = next(iter(groups.values()))
                _, missing = expand_shards(shards[0]) if shards else ([], ["?"])
                if missing:
                    info.notes.append(f"Missing shard(s): {', '.join(missing)}")
                    return info
                cls._inspect_gguf_set(shards, info, strata_root, data_dir)
                return info
            if len(groups) > 1:
                info.format = "GGUF_MULTI"
                info.candidates = groups
                info.notes.append(f"{target_path} holds {len(groups)} models: choose one file.")
                return info

            # 5. Check if it is a Hugging Face SafeTensors model
            if (target_path / "config.json").exists():
                cls._inspect_hf_safetensors(target_path, info)
                return info

        info.notes.append("No valid GGUF model, SafeTensors model, or Strata Pack found.")
        return info

    @staticmethod
    def _ready_paths(dir_path: Path) -> Dict[str, Path]:
        """A ready-made pack folder (setup.py ready_paths): pack/index.txt, and the dense, PLE and embedding GGUFs
        (qwen-nvfp4-gptq-dense.gguf, ple-fp8.gguf, token-embd-bf16.gguf). Also when the pack/ folder itself is given."""
        base = dir_path.parent if (dir_path / "index.txt").exists() and dir_path.name == "pack" else dir_path
        pack = base / "pack"
        if not (pack / "index.txt").exists():
            return {}
        ggufs = [f for f in base.glob("*.gguf") if not is_aux_gguf(f)]
        dense = [f for f in ggufs if "dense" in f.name.lower()]
        ple = [f for f in ggufs if f.name.lower().startswith("ple")]
        embd = [f for f in ggufs if "embd" in f.name.lower()]
        if len(dense) == 1 and len(ple) == 1 and len(embd) == 1:
            return {"pack": pack, "dense": dense[0], "ple": ple[0], "embd": embd[0]}
        return {}

    @classmethod
    def _inspect_ready(cls, ready: Dict[str, Path], info: ModelInfo, strata_root: Optional[Path],
                       data_dir: Optional[Path]) -> None:
        info.format = "READY_PACK"
        info.is_valid = True
        info.ready = ready
        info.pack_dir = ready["pack"]
        info.native_shard = ready["dense"]
        info.ple_shard = ready["ple"]
        info.gguf_shards = [ready["dense"]]
        info.total_size_bytes = sum(p.stat().st_size for p in ready["pack"].glob("*.bin") if p.is_file())
        cls._read_gguf_metadata(ready["dense"], info)
        info.family = "nvfp4"
        info.mmproj_file = cls._find_mmproj(info, data_dir)

    @classmethod
    def _inspect_gguf_set(cls, shards: List[Path], info: ModelInfo, strata_root: Optional[Path],
                          data_dir: Optional[Path]) -> None:
        info.format = "GGUF"
        info.is_valid = True
        info.gguf_shards = list(shards)
        info.total_size_bytes = sum(s.stat().st_size for s in shards if s.exists())
        info.native_shard = shards[0]
        info.ple_shard = find_ple_shard(shards, strata_root)
        if info.ple_shard is not None:
            tensors = gguf_tensors(info.ple_shard, strata_root) or {}
            info.ple_bytes = tensors.get(PLE_TENSOR, 0)
        else:
            info.notes.append(f"No shard has the {PLE_TENSOR} tensor: is this a Qwen3.8-Flash-Next GGUF?")
        cls._read_gguf_metadata(info.native_shard, info)
        info.family = detect_family(info)
        info.is_coder = info.family == "coder"
        info.mmproj_file = cls._find_mmproj(info, data_dir)

    @staticmethod
    def _find_mmproj(info: ModelInfo, data_dir: Optional[Path]) -> Optional[Path]:
        """The vision encoder: beside the model, else in the models folder (where setup.py downloads it), preferring
        the family's own file (mmproj-Swift-... for Swift)."""
        dirs = []
        if info.native_shard:
            dirs.append(info.native_shard.parent)
        if data_dir:
            dirs.append(data_dir / "models")
        found: List[Path] = []
        for d in dirs:
            if d.is_dir():
                found += sorted(f for f in d.glob("*.gguf") if "mmproj" in f.name.lower())
        if not found:
            return None
        want = "swift" if info.family == "swift" else None
        for f in found:
            if (want and want in f.name.lower()) or (not want and "swift" not in f.name.lower()):
                return f
        return found[0]

    @classmethod
    def _read_gguf_metadata(cls, gguf_path: Path, info: ModelInfo) -> None:
        try:
            meta = GGUFReader.read_metadata(gguf_path)
            arch = meta.get("general.architecture", "qwen4exp")
            info.architecture = str(arch)
            info.model_name = str(meta.get("general.name", gguf_path.stem))

            ftype = meta.get("general.file_type", None)
            info.quantization = cls._decode_gguf_file_type(ftype, gguf_path.name)

            prefix = f"{arch}."
            info.num_layers = int(meta.get(f"{prefix}block_count", 48))
            info.hidden_size = int(meta.get(f"{prefix}embedding_length", 4096))
            info.num_heads = int(meta.get(f"{prefix}attention.head_count", 32) or 32) \
                if not isinstance(meta.get(f"{prefix}attention.head_count"), (str, list)) else 32
            info.num_kv_heads = int(meta.get(f"{prefix}attention.head_count_kv", 8) or 8) \
                if not isinstance(meta.get(f"{prefix}attention.head_count_kv"), (str, list)) else 8
            info.max_context = int(meta.get(f"{prefix}context_length", TRAINED_CONTEXT_LIMIT))

            exp_count = meta.get(f"{prefix}expert_count", 0)
            if exp_count and int(exp_count) > 0:
                info.is_moe = True
                info.num_experts = int(exp_count)
                info.active_experts_per_tok = int(meta.get(f"{prefix}expert_used_count", 0) or 0)
        except Exception as e:
            info.notes.append(f"Partial error reading GGUF: {e}")

    @classmethod
    def _inspect_strata_pack(cls, dir_path: Path, info: ModelInfo, strata_root: Optional[Path],
                             data_dir: Optional[Path]) -> None:
        info.format = "STRATA_PACK"
        info.pack_dir = dir_path

        # 1. The GGUF the pack was built from: this script's record, else the shard name iq_pack.py wrote
        shard1: Optional[Path] = None
        src = read_pack_source(dir_path)
        if src and src.get("native") and Path(src["native"]).exists():
            shard1 = Path(src["native"])
        else:
            name, _ = read_pack_header(dir_path)
            if name:
                roots = []
                if data_dir:
                    roots.append(data_dir / "models")
                roots += [dir_path.parent.parent / "models", dir_path.parent / "models"]
                hits: List[Path] = []
                for r in dict.fromkeys(roots):
                    if r.is_dir():
                        hits += [h for h in r.rglob(name) if h.is_file()]
                hits = list(dict.fromkeys(hits))
                if len(hits) == 1:
                    shard1 = hits[0]
                elif len(hits) > 1:
                    info.notes.append(f"The pack was built from '{name}', and {len(hits)} files have that name: "
                                      + ", ".join(str(h.parent.name) for h in hits)
                                      + ". Give the GGUF itself with --model-dir.")
                else:
                    info.notes.append(f"The pack was built from '{name}', which is not under {', '.join(map(str, roots))}.")
            else:
                info.notes.append("The pack's native_experts.txt does not name its GGUF.")

        if shard1 is None:
            return
        shards, missing = expand_shards(shard1)
        if missing:
            info.notes.append(f"Missing shard(s) beside {shard1.name}: {', '.join(missing)}")
            return
        cls._inspect_gguf_set(shards, info, strata_root, data_dir)
        info.format = "STRATA_PACK"
        info.pack_dir = dir_path

    @classmethod
    def _inspect_hf_safetensors(cls, dir_path: Path, info: ModelInfo) -> None:
        info.format = "SAFETENSORS"
        cfg_file = dir_path / "config.json"
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            info.is_valid = True
            arch_list = cfg.get("architectures", [])
            info.architecture = arch_list[0] if arch_list else cfg.get("model_type", "Unknown")
            info.model_name = cfg.get("_name_or_path", dir_path.name)
            info.num_layers = cfg.get("num_hidden_layers", 48)
            info.hidden_size = cfg.get("hidden_size", 4096)
            info.num_heads = cfg.get("num_attention_heads", 32)
            info.num_kv_heads = cfg.get("num_key_value_heads", 8)
            info.max_context = cfg.get("max_position_embeddings", TRAINED_CONTEXT_LIMIT)

            if "num_experts" in cfg or "n_routed_experts" in cfg:
                info.is_moe = True
                info.num_experts = cfg.get("num_experts", cfg.get("n_routed_experts", 0))
                info.active_experts_per_tok = cfg.get("num_experts_per_tok", 0)

            total_bytes = sum(f.stat().st_size for f in dir_path.glob("*.safetensors"))
            if total_bytes == 0:
                total_bytes = sum(f.stat().st_size for f in dir_path.glob("*.bin"))
            info.total_size_bytes = total_bytes
            info.quantization = "BF16 / FP16 (Unquantized)"

        except Exception as e:
            info.notes.append(f"Error parsing Hugging Face config.json: {e}")

    @staticmethod
    def _decode_gguf_file_type(ftype: Optional[int], filename: str) -> str:
        mapping = {
            0: "ALL_F32", 1: "MOSTLY_F16", 2: "MOSTLY_Q4_0", 3: "MOSTLY_Q4_1",
            7: "MOSTLY_Q8_0", 8: "MOSTLY_Q5_0", 9: "MOSTLY_Q5_1",
            10: "MOSTLY_Q2_K", 11: "MOSTLY_Q3_K_S", 12: "MOSTLY_Q3_K_M",
            13: "MOSTLY_Q3_K_L", 14: "MOSTLY_Q4_K_S", 15: "MOSTLY_Q4_K_M",
            16: "MOSTLY_Q5_K_S", 17: "MOSTLY_Q5_K_M", 18: "MOSTLY_Q6_K",
            19: "MOSTLY_IQ2_XXS", 20: "MOSTLY_IQ2_XS", 21: "MOSTLY_Q2_K_S",
            22: "MOSTLY_IQ3_XS", 23: "MOSTLY_IQ3_XXS", 24: "MOSTLY_IQ1_S",
            25: "MOSTLY_IQ4_NL", 26: "MOSTLY_IQ3_S", 27: "MOSTLY_IQ3_M",
            28: "MOSTLY_IQ2_S", 29: "MOSTLY_IQ2_M", 30: "MOSTLY_IQ4_XS",
            31: "MOSTLY_IQ1_M", 32: "MOSTLY_BF16",
        }
        if ftype in mapping:
            return mapping[ftype]

        name_upper = filename.upper()
        for q in ["IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ3_S", "Q4_K_M", "Q4_0", "Q8_0", "BF16", "FP16"]:
            if q in name_upper:
                return q
        return "GGUF_CUSTOM"


def choose_candidate(info: ModelInfo, auto_confirm: bool) -> Optional[Path]:
    """A folder with several models: the shard 1 of the one the user picks (None under --yes: it must be named)."""
    names = sorted(info.candidates)
    print(color_text(f"\n[!] {info.source_path} holds {len(names)} models:", TermColor.YELLOW))
    for i, n in enumerate(names, 1):
        shards = info.candidates[n]
        size = sum(s.stat().st_size for s in shards if s.exists()) / 1024 ** 3
        print(f"  [{i}] {n} ({len(shards)} file(s), {size:.1f} GB)")
    if auto_confirm:
        print(color_text("Pass the model file itself with --model-dir (--yes does not pick one).", TermColor.RED))
        return None
    sel = safe_input("Select model number: ", default="")
    try:
        idx = int(sel) - 1
        if 0 <= idx < len(names):
            return info.candidates[names[idx]][0]
    except ValueError:
        pass
    print(color_text("Invalid selection.", TermColor.RED))
    return None


# ==============================================================================
# 4. STRATA ENVIRONMENT AND TOOLS RESOLUTION
# ==============================================================================

def _load_strata_settings() -> Dict[str, Any]:
    for cfg_file in (Path.home() / ".config" / "strata" / STRATA_CONFIG_FILENAME,
                     Path(os.environ.get("APPDATA", "")) / "Strata" / STRATA_CONFIG_FILENAME):
        if cfg_file.is_file():
            try:
                data = json.loads(cfg_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (OSError, ValueError):
                pass
    return {}


class StrataEnvironment:
    """Locates the Strata repository, Python virtual environment, and official tools."""

    @staticmethod
    def _is_root(p: Path) -> bool:
        return p.exists() and (p / "serve" / "server.py").exists()

    @classmethod
    def find_strata_root(cls, custom_root: Optional[str] = None) -> Optional[Path]:
        if custom_root:
            p = Path(custom_root).expanduser().resolve()
            if cls._is_root(p):
                return p

        env_root = os.environ.get("STRATA_ROOT") or os.environ.get("STRATA_HOME")
        if env_root:
            p = Path(env_root).resolve()
            if cls._is_root(p):
                return p

        # 1. The folder this script lives in, then the default install
        for cand in (SCRIPT_DIR, DEFAULT_STRATA_ROOT):
            if cls._is_root(cand):
                return cand.resolve()

        # 2. Check settings.json
        for inst in _load_strata_settings().get("installs", []):
            p_inst = Path(inst).resolve()
            if cls._is_root(p_inst):
                return p_inst

        # 3. Check current directory and subdirectories
        for cand in [Path.cwd(), Path.cwd() / "Strata", Path.cwd() / "strata", Path.home() / "Strata"]:
            if cls._is_root(cand):
                return cand.resolve()

        return None

    @classmethod
    def find_strata_data_dir(cls, custom_data: Optional[str] = None,
                             hint_path: Optional[Path] = None) -> Optional[Path]:
        """The Strata-data folder (models/, packs/, mtp/): --data-dir, $STRATA_DATA_DIR, the folder the given model
        or pack sits in (<data>/models/<name>/file.gguf, <data>/packs/<name>), then the known places."""
        # Paths are kept as given, links not resolved: the config names the folders the way setup.py does
        # (bin/Strata-data/... stays bin/Strata-data/..., not the folder the link points to)
        if custom_data:
            p = Path(os.path.abspath(os.path.expanduser(custom_data)))
            if p.exists():
                return p

        env_data = os.environ.get("STRATA_DATA_DIR")
        if env_data:
            p = Path(os.path.abspath(env_data))
            if p.exists():
                return p

        if hint_path:
            p = Path(os.path.abspath(str(hint_path)))
            # <data>/models/<x>/file.gguf, <data>/packs/<x>, <data>/artifacts[/<x>] (artifacts/ itself may be a pack)
            for parent in [p, *p.parents]:
                if parent.name in ("models", "packs", "artifacts") and parent != p and parent.parent.exists():
                    return parent.parent
                if parent.name == "artifacts" and parent == p and parent.parent.exists() and \
                        (parent.parent / "models").is_dir():
                    return parent.parent
            # else the nearest folder that looks like a Strata-data folder (models/ beside mtp/, packs/ or artifacts/)
            for parent in p.parents:
                if (parent / "models").is_dir() and any((parent / d).is_dir() for d in ("mtp", "packs", "artifacts")):
                    return parent

        candidates = [DEFAULT_DATA_DIR]
        settings_dir = _load_strata_settings().get("data_dir")
        if settings_dir:
            candidates.append(Path(settings_dir))
        candidates += [Path.cwd() / "Strata-data"]
        for c in candidates:
            if c.exists():
                return Path(os.path.abspath(str(c)))

        return None

    @classmethod
    def find_strata_python(cls, strata_root: Optional[Path] = None) -> str:
        if strata_root:
            venv_py = strata_root / ".venv" / "bin" / "python"
            if venv_py.exists():
                return str(venv_py)
            venv_py_win = strata_root / ".venv" / "Scripts" / "python.exe"
            if venv_py_win.exists():
                return str(venv_py_win)
        return sys.executable

    @classmethod
    def tool_env(cls, strata_root: Optional[Path]) -> Dict[str, str]:
        """The tools' environment as setup.py step 6 gives it: STRATA_GGUF_PY at the llama.cpp it cloned.  Unbuffered,
        so a tool's printed lines and its progress lines (stderr) reach the terminal in the order it wrote them."""
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        if strata_root:
            gguf_py = strata_root / "third_party" / "llama.cpp" / "gguf-py"
            if gguf_py.is_dir() and not env.get("STRATA_GGUF_PY"):
                env["STRATA_GGUF_PY"] = str(gguf_py)
        return env

    @classmethod
    def find_tool(cls, tool_name: str, strata_root: Optional[Path] = None) -> Optional[Path]:
        if not strata_root:
            strata_root = cls.find_strata_root()

        if strata_root:
            cand = strata_root / "tools" / tool_name
            if cand.exists():
                return cand.resolve()
            cand_direct = strata_root / tool_name
            if cand_direct.exists():
                return cand_direct.resolve()

        # Check in CWD
        cand_cwd = Path.cwd() / "tools" / tool_name
        if cand_cwd.exists():
            return cand_cwd.resolve()

        return None

    @classmethod
    def find_mtp_runtime(cls, data_dir: Optional[Path] = None) -> Optional[Path]:
        """<data>/mtp/rt when tools/mtp_rt.py finished there (setup.py step 6 tests rt/experts.bin; mtp_rt.py writes
        experts.bin, then dense.bin)."""
        if data_dir:
            rt = data_dir / "mtp" / "rt"
            if (rt / "experts.bin").exists() and (rt / "dense.bin").exists():
                return rt
        return None

    @classmethod
    def find_expert_profile(cls, strata_root: Optional[Path], num_experts: int = 0,
                            is_coder: bool = False) -> Tuple[Optional[Path], str]:
        """The data/expert-profile*.bin whose header has this model's experts per layer (512: the original and its
        fine-tunes, 256: the Coder), as serve/server.py profile_shape reads it. (path, how it was chosen)."""
        if not strata_root:
            return None, "no Strata folder"
        files = [strata_root / "data" / "expert-profile.bin", strata_root / "data" / "expert-profile-coder.bin"]
        files = [f for f in files if f.exists()]
        if num_experts:
            for f in files:
                shape = profile_shape(f)
                if shape and shape[1] == num_experts:
                    return f, f"header: {shape[0]} layers x {shape[1]} experts"
            return None, f"no data/expert-profile*.bin has {num_experts} experts per layer"
        name = "expert-profile-coder.bin" if is_coder else "expert-profile.bin"
        f = strata_root / "data" / name
        return (f if f.exists() else None), "by family (the GGUF does not say its expert count)"

    @classmethod
    def find_engine_binary(cls, strata_root: Optional[Path] = None, name: str = "strata") -> Optional[Path]:
        """engine/<name> (what setup.py writes into "exe"), else build/<name>."""
        if not strata_root:
            strata_root = cls.find_strata_root()
        exe = name + (".exe" if WIN else "")
        if strata_root:
            for cand in (strata_root / "engine" / exe, strata_root / "build" / exe):
                if cand.is_file():
                    return cand.resolve()
        return None

    @classmethod
    def build_meta(cls, strata_root: Optional[Path]) -> Dict[str, Any]:
        if strata_root:
            try:
                data = json.loads((strata_root / "engine" / "BUILD.json").read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (OSError, ValueError):
                pass
        return {}

    @classmethod
    def find_cuda_lib_dirs(cls, strata_root: Optional[Path] = None) -> List[str]:
        """setup.py: meta.get("lib_dirs") or meta.get("cuda_dirs") from engine/BUILD.json, the ones that exist."""
        meta = cls.build_meta(strata_root or cls.find_strata_root())
        dirs = [str(d) for d in (meta.get("lib_dirs") or meta.get("cuda_dirs") or []) if Path(d).is_dir()]
        if not dirs:
            dirs = [d for d in ("/usr/local/cuda/bin", "/usr/local/cuda/lib64") if Path(d).is_dir()]
        return dirs


def profile_shape(path: Path) -> Optional[Tuple[int, int]]:
    """An expert profile's (layers, experts per layer) from its header (serve/server.py profile_shape)."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
            size = os.fstat(f.fileno()).st_size
    except OSError:
        return None
    if len(head) < 24 or head[:4] != b"STRP":
        return None
    _, nl, ne, _, n = struct.unpack("<5I", head[4:])
    return (nl, ne) if size >= 24 + 4 * n else None


def resolve_rope(ctx: int, scaling: Optional[str], scale: Optional[float],
                 trained: int = TRAINED_CONTEXT_LIMIT) -> Tuple[Optional[str], Optional[float]]:
    """setup.py resolve_rope: (scaling, scale) for the served context; (None, None) = no rope flags. Past the trained
    262144 an omitted method is yarn and the factor is context / 262144; an explicit 'none' there is refused."""
    derived = max(1.0, float(ctx) / float(trained))
    if ctx <= trained:
        if scale is not None and scaling in (None, "none"):
            raise ValueError("--rope-scale needs --rope-scaling linear or yarn (the chosen context fits the "
                             "trained 262144, so there is nothing to scale)")
        if scaling in (None, "none"):
            return None, None
        return scaling, scale if scale is not None else derived
    if scaling == "none":
        raise ValueError(f"a {ctx // 1024}K context is past the model's trained 262144, and --rope-scaling none "
                         "keeps the stock angles there. Pick --rope-scaling yarn or linear, or a context of 262144 "
                         "or lower")
    return scaling or "yarn", scale if scale is not None else derived


def kv_ram_gb(ctx: int, kv: str) -> float:
    """The RAM a streamed KV cache takes (setup.py kv_streaming_ram_gb): 13 attention layers x bytes per cell."""
    return ctx * KV_ATTN_LAYERS * KV_CELL_BYTES.get(kv, 1056) / 1e9


def recommend_pool_workers(args: List[str], strata_root: Optional[Path]) -> Tuple[List[str], str]:
    """setup.py recommend_pool_workers: on a hybrid CPU with more E-cores than P-cores, P - 1 + E // 2 workers
    (#642's measurement); on any other CPU no flag (the engine's own count: one per physical core but the host's)."""
    if "--pool-workers" in args:
        return args, "kept (calibration)"
    setup = load_setup(strata_root)
    cores = None
    if setup is not None:
        try:
            cores = setup.cpu_cores()
        except Exception:
            cores = None
    if not cores:
        return args, "engine default (not a hybrid CPU, or the OS does not say)"
    p, e = cores
    if e <= p:
        return args, f"engine default ({p} P + {e} E cores: not more E- than P-cores)"
    n = max(1, p - 1 + e // 2)
    return [*args, "--pool-workers", str(n)], f"hybrid CPU {p} P + {e} E cores: {p} - 1 + {e} // 2 = {n}"


def apply_calibration(cfg: Dict[str, Any], strata_root: Optional[Path]) -> Optional[str]:
    """The settings `./setup.sh --calibrate` measured for exactly this PC, model name, context and images setting
    (setup.py hardware_key), applied the way setup.py does (tools/calibrate.py apply / apply_env)."""
    setup = load_setup(strata_root)
    if setup is None:
        return None
    try:
        cal = (setup.load_settings().get("calibration") or {}).get(setup.hardware_key(cfg))
    except Exception:
        return None
    if not cal:
        return None
    try:
        sys.path.insert(0, str(strata_root / "tools"))
        import calibrate as CAL
        cfg["args"] = CAL.apply(cfg["args"], cal.get("settings") or {})
        if "env" in cal:
            cfg["env"] = CAL.apply_env(cfg.get("env"), cal["env"])
    except Exception as e:
        print(color_text(f"  [!] the saved calibration could not be applied ({e})", TermColor.YELLOW))
        return None
    return cal.get("date") or "saved"


def engine_flag_problems(exe: Path, args: List[str]) -> List[str]:
    """The options in `args` the engine binary does not know: every flag the engine parses is a string literal in it
    ("--kv\\0"), and the engine stops at an unknown one ("unknown argument", exit code 2: EngineDied)."""
    try:
        blob = exe.read_bytes()
    except OSError:
        return []
    return [a for a in args if a.startswith("--") and (a.encode() + b"\0") not in blob]


# ==============================================================================
# 5. MEMORY AND RESOURCE ESTIMATION
# ==============================================================================

class MemoryPlan:
    def __init__(self):
        self.is_viable: bool = True
        self.requested_context: int = 0
        self.allocated_context: int = 0
        self.kv_quant: str = DEFAULT_KV_QUANT
        self.kv_resident_tokens: int = 0           # 0: no KV streaming (the whole KV cache in VRAM)
        self.kv_bytes_per_token: int = 0
        self.kv_total_gb: float = 0.0
        self.kv_stream_ram_gb: float = 0.0
        self.model_ram_gb: float = 0.0
        self.ram_total_gb: float = 0.0
        self.unsloth_budget_gib: Optional[int] = None
        self.warning_messages: List[str] = []
        self.recommendations: List[str] = []


class MemoryEstimator:
    """setup.py's RAM rules: the experts live in RAM (an arena) beside ~12 GB for the rest; from 64K up the KV cache is
    streamed from RAM when it fits there too (--kv-resident 32768). The VRAM side is the engine's: it sizes the
    expert cache to the free VRAM itself (--expert-cache auto)."""

    @classmethod
    def evaluate(
        cls,
        hw: HardwareInfo,
        model: ModelInfo,
        requested_context: int,
        preferred_kv_quant: str = DEFAULT_KV_QUANT,
        kv_streaming: str = "auto",
    ) -> MemoryPlan:
        plan = MemoryPlan()
        plan.requested_context = requested_context
        plan.allocated_context = requested_context
        plan.kv_quant = preferred_kv_quant
        plan.kv_bytes_per_token = KV_ATTN_LAYERS * KV_CELL_BYTES.get(preferred_kv_quant, 1056)
        plan.kv_total_gb = requested_context * plan.kv_bytes_per_token / 1e9
        plan.ram_total_gb = hw.total_ram_bytes / 1024 ** 3   # setup.py ram_gb()
        plan.kv_stream_ram_gb = kv_ram_gb(requested_context, preferred_kv_quant)
        plan.model_ram_gb = model.experts_gb + MODEL_RAM_BESIDE_EXPERTS_GB

        if not hw.gpus:
            plan.is_viable = False
            plan.warning_messages.append("No compatible NVIDIA CUDA or AMD ROCm GPU detected.")

        # Unsloth's files: every expert in RAM where they fit with 16 GB beside them, else a RAM budget
        budget_mode = model.family == "unsloth" and model.experts_gb + UNSLOTH_RESIDENT_BESIDE_GB > plan.ram_total_gb

        stream_fits = plan.ram_total_gb >= plan.model_ram_gb + plan.kv_stream_ram_gb + 1
        if requested_context >= KV_STREAM_MIN_CTX and kv_streaming != "off" and (stream_fits or kv_streaming == "on"):
            plan.kv_resident_tokens = KV_RESIDENT_TOKENS
            if not stream_fits:
                plan.warning_messages.append(
                    f"KV streaming needs ~{plan.kv_stream_ram_gb:.1f} GB of RAM beside the ~{plan.model_ram_gb:.0f} GB "
                    f"the model uses; this PC has {plan.ram_total_gb:.0f} GB (kept: --kv-streaming on).")
        elif requested_context >= KV_STREAM_MIN_CTX and kv_streaming != "off":
            plan.recommendations.append(
                f"KV streaming off: it needs ~{plan.kv_stream_ram_gb:.1f} GB of RAM beside the ~{plan.model_ram_gb:.0f} "
                f"GB the model uses; the KV cache ({plan.kv_total_gb:.1f} GB) stays in VRAM (fewer cached experts).")

        if budget_mode:
            kv_in_ram = plan.kv_stream_ram_gb if plan.kv_resident_tokens else 0.0
            gib = round(hw.total_ram_bytes / 1024 ** 3) - UNSLOTH_RAM_LEFT_GB - math.ceil(kv_in_ram)
            plan.unsloth_budget_gib = max(8, min(gib, int(model.experts_gb)))
            plan.recommendations.append(
                f"Unsloth RAM budget: {plan.unsloth_budget_gib} GiB of the {model.experts_gb:.0f} GB of experts kept in "
                "RAM (the most used), the rest read from the GGUF (--resident-budget-gib).")
        elif model.format != "READY_PACK" and plan.model_ram_gb > plan.ram_total_gb:
            plan.is_viable = False
            plan.warning_messages.append(
                f"The model's experts ({model.experts_gb:.0f} GB) and ~{MODEL_RAM_BESIDE_EXPERTS_GB} GB beside them do "
                f"not fit this PC's {plan.ram_total_gb:.0f} GB of RAM.")
            plan.recommendations.append("Use the low-RAM mode: --convert-model --experts-bin and --low-ram mmap "
                                        "(or resident), as setup.py does.")
        return plan


# ==============================================================================
# 6. MODEL CONVERSION AND ARTIFACT GENERATION (--convert-model)
# ==============================================================================

def run_streamed(cmd: List[str], env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    """Run a tool, showing its output as it comes, and keep the output for the checks after it."""
    print(f"Command: {' '.join(cmd)}")
    out: List[str] = []
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                                bufsize=1)
    except OSError as e:
        return 127, str(e)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        out.append(line)
    return proc.wait(), "".join(out)


class ModelArtifacts:
    """One model's own artifacts, apart from every other model's:
        <artifacts>/<name>/packs     the pack the engine reads (--pack; tokenizer/ in it)
        <artifacts>/<name>/mtp       the MTP tensors and mtp-q2_0.gguf the draft layer is built from
        <artifacts>/<name>/mtp/rt    the MTP draft layer (--mtp), with this model's own draft_vocab.bin
    and <name>/model-config-source.json: the GGUF the folder belongs to."""

    def __init__(self, root: Path, name: str):
        self.root = root
        self.name = name
        self.base = root / name
        self.packs = self.base / "packs"
        self.mtp = self.base / "mtp"
        self.rt = self.mtp / "rt"
        self.record = self.base / PACK_SOURCE_FILE

    def create(self) -> None:
        for d in (self.packs, self.rt):
            d.mkdir(parents=True, exist_ok=True)

    def owner(self) -> Optional[str]:
        """The GGUF (shard 1, or a ready pack's dense GGUF) this folder was made for, or None for a free folder."""
        for f in (self.record, self.packs / PACK_SOURCE_FILE):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(rec, dict) and rec.get("native"):
                    return str(rec["native"])
            except (OSError, ValueError):
                pass
        return None

    def rt_complete(self) -> bool:
        return (self.rt / "experts.bin").exists() and (self.rt / "dense.bin").exists()

    def write_record(self, model_info: "ModelInfo") -> None:
        """Which GGUF this folder (and its pack) belongs to: absolute paths and sizes, so a model whose files have
        the same names (Swift and Huihui's Swift, Qwen and Coder) never shares it."""
        rec = {"native": str(model_info.native_shard.absolute()),
               "shards": [str(s.absolute()) for s in model_info.gguf_shards],
               "sizes": [s.stat().st_size for s in model_info.gguf_shards if s.exists()],
               "family": model_info.family,
               "written": time.strftime("%Y-%m-%d %H:%M:%S")}
        text = json.dumps(rec, indent=2) + "\n"
        for f in (self.record, self.packs / PACK_SOURCE_FILE):
            try:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(text, encoding="utf-8")
            except OSError:
                pass

    @staticmethod
    def root_for(data_dir: Optional[Path], artifacts_dir: Optional[Union[Path, str]] = None) -> Optional[Path]:
        """The artifacts folder: --artifacts-dir, else <data>/artifacts; created when missing."""
        if artifacts_dir:
            root = Path(os.path.abspath(os.path.expanduser(str(artifacts_dir))))
        elif data_dir:
            root = data_dir / "artifacts"
        else:
            return None
        root.mkdir(parents=True, exist_ok=True)
        return root

    @classmethod
    def for_model(cls, model_info: "ModelInfo", root: Path, name: Optional[str] = None) -> "ModelArtifacts":
        """The model's folder: <model name, lower case>, or <its folder>__<name> when that one belongs to another
        GGUF of the same file name; --artifacts-name chooses it."""
        if name:
            return cls(root, name)
        src = model_info.native_shard
        base = (model_info.ready["dense"].parent.name if model_info.ready else gguf_base_name(src)).lower()
        names = [base, f"{src.parent.name.lower()}__{base}",
                 f"{base}-{hashlib.sha1(str(src.absolute()).encode()).hexdigest()[:8]}"]
        for n in names:
            m = cls(root, n)
            own = m.owner()
            if own is None or same_file(own, src):
                return m
        return cls(root, names[-1])


def copy_tree(src: Path, dst: Path) -> Tuple[int, int]:
    """dst/ as a full copy of src/: every file its own copy (no links), so nothing a model's folder holds is shared
    with another folder.  Files already there are kept.  Refused before copying anything when the disk lacks the room
    (the bytes to copy and 1 GiB beside them).  (files copied, bytes copied)."""
    todo = [(f, dst / f.relative_to(src)) for f in sorted(src.rglob("*")) if f.is_file()]
    todo = [(f, t) for f, t in todo if not t.exists()]
    need = sum(f.stat().st_size for f, _ in todo)
    dst.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(str(dst)).free
    if need + (1 << 30) > free:
        raise OSError(f"copying {src} needs {need / 1024 ** 3:.1f} GiB and {dst} has {free / 1024 ** 3:.1f} GiB free")
    for f, t in todo:
        t.parent.mkdir(parents=True, exist_ok=True)
        tmp = t.with_name(t.name + ".tmp")
        shutil.copy2(f, tmp)                        # written whole, then named: a stopped copy leaves no half file
        os.replace(tmp, t)
    return len(todo), need


def make_independent(folder: Path) -> Tuple[int, int]:
    """Every file in `folder` its own copy: a file that shares its bytes with another name (a hard link from an
    earlier version of this script) is copied and the copy put in its place; the other name keeps the original.
    (files made independent, bytes copied)."""
    n = size = 0
    if not folder.is_dir():
        return 0, 0
    for f in sorted(folder.rglob("*")):
        try:
            st = f.lstat()
        except OSError:
            continue
        if not f.is_file() or f.is_symlink() or st.st_nlink <= 1:
            continue
        tmp = f.with_name(f.name + ".tmp")
        shutil.copy2(f, tmp)
        os.replace(tmp, f)
        n += 1
        size += st.st_size
    return n, size


class ArtifactGenerator:
    """Builds a model's artifacts in <artifacts>/<model>/: the pack (tools/iq_pack.py) and the MTP draft layer."""

    @classmethod
    def generate_artifacts(
        cls,
        source_model_dir: Path,
        compat_bf16: str = "auto",
        experts_bin: bool = False,
        build_mtp: bool = False,
        draft_vocab: Optional[str] = None,
        overwrite: bool = False,
        auto_confirm: bool = False,
        strata_root: Optional[str] = None,
        data_dir: Optional[str] = None,
        artifacts_dir: Optional[Path] = None,
        artifacts_name: Optional[str] = None,
    ) -> Tuple[bool, Optional[Path]]:
        """(done, the model's GGUF shard 1 or ready-pack folder, for the config step)."""
        strata_path = StrataEnvironment.find_strata_root(strata_root)
        data_dir_path = StrataEnvironment.find_strata_data_dir(data_dir, source_model_dir)

        print(color_text("\n[1/3] Validating source model directory and format...", TermColor.CYAN))
        model_info = ModelAnalyzer.inspect(source_model_dir, strata_path, data_dir_path)
        if model_info.format == "GGUF_MULTI":
            pick = choose_candidate(model_info, auto_confirm)
            if pick is None:
                return False, None
            model_info = ModelAnalyzer.inspect(pick, strata_path, data_dir_path)

        if not model_info.is_valid:
            print(color_text(f"Error: Directory does not contain a valid model: {source_model_dir}", TermColor.RED))
            for note in model_info.notes:
                print(f" - {note}")
            return False, None

        print(f" - Detected format: {color_text(model_info.format, TermColor.GREEN)} (family: {model_info.family})")
        print(f" - Architecture: {model_info.architecture}")
        print(f" - Total size on disk: {model_info.total_size_gb:.2f} GB")
        print(f" - Quantization: {model_info.quantization}")
        if model_info.native_shard:
            print(f" - Primary shard (--native): {model_info.native_shard.name}")
        if model_info.ple_shard:
            print(f" - PLE table shard ({PLE_TENSOR}): {model_info.ple_shard.name}")

        if model_info.format == "SAFETENSORS":
            print(color_text("\n[2/3] Model in SafeTensors format (Original Hugging Face)", TermColor.YELLOW))
            print("Strata reads Qwen3.8-Flash-Next GGUF files; convert this model to GGUF with llama.cpp first:")
            print("  git clone https://github.com/ggerganov/llama.cpp")
            print("  pip install -r llama.cpp/requirements.txt")
            print(f"  python3 llama.cpp/convert_hf_to_gguf.py {source_model_dir} --outfile <models>/<name>/<name>.gguf")
            return False, None

        if model_info.ple_shard is None or model_info.native_shard is None:
            print(color_text(f"Error: no shard has the {PLE_TENSOR} tensor (is this a Qwen3.8-Flash-Next GGUF?).",
                             TermColor.RED))
            return False, None

        # Every model file whole before anything is built from it (setup.py check_shards), and marked done after
        model_files = [model_info.ready[k] for k in ("dense", "ple", "embd")] if model_info.ready else \
            list(model_info.gguf_shards)
        problems = [p for p in (gguf_problem(f, strata_path) for f in model_files) if p]
        if problems:
            for p in problems:
                print(color_text(f"Error: {p}", TermColor.RED))
            print("Download the file again (delete it and its .done mark first).")
            return False, None
        print(f" - Model files whole (header vs size): {len(model_files)} [OK]")

        # The model's own artifacts folder: <artifacts>/<model>/{packs, mtp, mtp/rt}, created when missing
        root = ModelArtifacts.root_for(data_dir_path, artifacts_dir)
        if root is None:
            print(color_text("Error: no Strata-data folder found; pass --artifacts-dir or --data-dir.", TermColor.RED))
            return False, None
        m = ModelArtifacts.for_model(model_info, root, artifacts_name)
        own = m.owner()
        if own and not same_file(own, model_info.native_shard):
            print(color_text(f"Error: {m.base} belongs to {own}; choose another --artifacts-name.", TermColor.RED))
            return False, None
        m.create()
        print(f" - Model artifacts folder: {m.base}")
        print(f"     packs/   {m.packs}")
        print(f"     mtp/rt/  {m.rt}")
        result_path = model_info.ready["pack"].parent if model_info.ready else model_info.native_shard

        if model_info.format == "READY_PACK":
            # a ready-made pack comes with the download: packs/ gets its own full copy of it
            print(color_text("\n[2/3] Ready-made pack: copying it into packs/ ...", TermColor.CYAN))
            try:
                n, size = copy_tree(model_info.ready["pack"], m.packs)
            except OSError as e:
                print(color_text(f"Error: {e}", TermColor.RED))
                return False, None
            print(f"  copied {n} file(s), {size / 1024 ** 3:.2f} GiB")
            m.write_record(model_info)
        elif pack_complete(m.packs) and pack_matches(m.packs, model_info.native_shard)[0] and not overwrite and \
                (not experts_bin or (m.packs / "experts.bin").exists()):
            print(color_text(f"\nThe pack in {m.packs} is already complete for this model: kept as it is "
                             "(--overwrite rebuilds it).", TermColor.GREEN))
        else:
            if not cls._pack(model_info, m, compat_bf16, experts_bin, strata_path):
                return False, None

        if strata_path:
            cls.ensure_model_mtp(m, data_dir_path, strata_path, build_mtp, draft_vocab, auto_confirm)

        # The processed model files' finish marks, as setup.py writes them: <file>.gguf.done
        written, kept = mark_done(model_files)
        for d in written:
            print(f" - Finish mark written: {d.name}")
        if kept:
            print(f" - Finish mark already there (kept): {', '.join(d.name for d in kept)}")
        return True, result_path

    @classmethod
    def _pack(cls, model_info: "ModelInfo", m: ModelArtifacts, compat_bf16: str, experts_bin: bool,
              strata_path: Optional[Path]) -> bool:
        pack_dir = m.packs
        strata_py = StrataEnvironment.find_strata_python(strata_path)
        iq_pack_tool = StrataEnvironment.find_tool("iq_pack.py", strata_path)
        env = StrataEnvironment.tool_env(strata_path)

        print(color_text("\n[2/3] Packing GGUF model using tools/iq_pack.py...", TermColor.CYAN))
        if not iq_pack_tool:
            print(color_text("Error: 'tools/iq_pack.py' not found. Specify '--strata-root /path/to/Strata'.", TermColor.RED))
            return False
        print(f"Python interpreter: {strata_py}")

        # --compat-bf16: setup.py passes it for Unsloth's files only (FAMILIES["unsloth"]["pack_args"]); in auto mode
        # a file iq_pack.py refuses without it ("... use --compat-bf16") is packed again with it
        use_compat = compat_bf16 == "on" or (compat_bf16 == "auto" and model_info.family == "unsloth")
        cmd = [strata_py, str(iq_pack_tool), "--gguf", str(model_info.native_shard), "--out", str(pack_dir)]
        rc, out = run_streamed(cmd + (["--compat-bf16"] if use_compat else []), env)
        if rc != 0 and not use_compat and compat_bf16 == "auto" and "--compat-bf16" in out:
            print(color_text("\n[i] iq_pack.py asks for --compat-bf16 for this file: packing again with it.",
                             TermColor.CYAN))
            use_compat = True
            rc, out = run_streamed(cmd + ["--compat-bf16"], env)
        if rc != 0:
            print(color_text(f"Error executing iq_pack.py (Code: {rc})", TermColor.RED))
            return False

        # The low-RAM mode's experts.bin: a second run, as setup.py does
        if experts_bin and not (pack_dir / "experts.bin").exists():
            print(color_text("\nWriting experts.bin for the low-RAM mode (one time, the size of the experts) ...",
                             TermColor.CYAN))
            rc, _ = run_streamed(cmd + (["--compat-bf16"] if use_compat else []) + ["--experts-bin"], env)
            if rc != 0:
                print(color_text(f"Error writing experts.bin (Code: {rc})", TermColor.RED))
                return False

        # Validation of generated artifacts
        print(color_text("\n[3/3] Validating accuracy and completeness of generated artifacts...", TermColor.CYAN))
        required_files = ["index.txt", "dense.bin", "native_experts.txt", "tokenizer/vocab.json",
                          "tokenizer/merges.txt", "tokenizer/token_type.json"]
        missing = [rf for rf in required_files if not (pack_dir / rf).exists()]
        if missing:
            print(color_text(f"Critical warning: Missing required pack files: {missing}", TermColor.RED))
            return False
        header_name, _ = read_pack_header(pack_dir)
        if header_name != model_info.native_shard.name:
            print(color_text(f"Error: native_experts.txt names '{header_name}', not {model_info.native_shard.name}.",
                             TermColor.RED))
            return False
        print(color_text(f" - native_experts.txt: built from {header_name} [OK]", TermColor.GREEN))
        if (pack_dir / "tokenizer" / "chat_template.jinja").exists():
            print(color_text(" - Official tokenizer (vocab.json, chat_template.jinja): Present", TermColor.GREEN))
        else:
            print(color_text(" - Tokenizer present; no chat_template.jinja (the server uses the original model's)",
                             TermColor.YELLOW))
        m.write_record(model_info)

        print(color_text("Artifacts verified successfully!", TermColor.GREEN))
        print("\nResulting files in destination:")
        for item in sorted(pack_dir.iterdir()):
            if item.is_file():
                print(f"  - {item.name:<25} ({human_size(item.stat().st_size)})")
            elif item.is_dir():
                sub_count = len(list(item.iterdir()))
                print(f"  - {item.name + '/':<25} ({sub_count} files)")
        return True

    @classmethod
    def find_mtp_source(cls, m: ModelArtifacts, data_dir: Optional[Path], strata_root: Path) -> Optional[Path]:
        """A finished, verified MTP folder to give this model its own copy from: another model's
        <artifacts>/<x>/mtp, else setup.py's <data>/mtp (the draft layer is the same for every model: it comes from
        the original checkpoint)."""
        cands = []
        if m.root.is_dir():
            cands += [d / "mtp" for d in sorted(m.root.iterdir()) if d.is_dir() and d != m.base]
        if data_dir:
            cands.append(data_dir / "mtp")
        for c in cands:
            if (c / "rt" / "experts.bin").exists() and (c / "rt" / "dense.bin").exists() and \
                    not cls.mtp_corrupt(c, strata_root):
                return c
        return None

    @classmethod
    def ensure_model_mtp(cls, m: ModelArtifacts, data_dir: Optional[Path], strata_root: Path, build_mtp: bool,
                         draft_vocab: Optional[str], auto_confirm: bool) -> Optional[Path]:
        """The model's own MTP draft layer in <artifacts>/<model>/mtp/rt (setup.py step 6): kept when it is there and
        its tensors are the checkpoint's; else a full copy of a finished, verified one (no download; ~6.5 GB of
        disk); else built with tools/mtp_*.py (~5 GB download) when asked.  Nothing in it is shared with another
        folder: files linked by an earlier version of this script are turned into copies first.  Then the model's draft-vocabulary subset: the one
        asked for (--draft-vocab), else the one already there (cjk when there is none)."""
        m.create()
        for folder in (m.mtp, m.packs):
            n, size = make_independent(folder)
            if n:
                print(f" - {folder}: {n} file(s) shared with another folder made independent "
                      f"({size / 1024 ** 3:.2f} GiB copied)")
        rebuild = False
        if m.rt_complete():
            if cls.mtp_corrupt(m.mtp, strata_root):
                print(color_text("[!] Some MTP tensors are not the checkpoint's (#327): fetching them again and "
                                 "rebuilding this model's draft layer.", TermColor.YELLOW))
                rebuild = True
        else:
            src = cls.find_mtp_source(m, data_dir, strata_root)
            if src is not None:
                print(f" - Copying the MTP draft layer from {src} (no download) ...")
                try:
                    n, size = copy_tree(src, m.mtp)
                    print(f" - MTP draft layer: {n} file(s), {size / 1024 ** 3:.2f} GiB copied into {m.mtp}")
                except OSError as e:
                    print(color_text(f"  [!] {e}", TermColor.YELLOW))
        if rebuild or not m.rt_complete():
            want = build_mtp or rebuild
            if not want:
                print(color_text(f"\n[MTP Check] No MTP draft layer in {m.mtp} and none to copy from.", TermColor.YELLOW))
                print("  setup.py always builds it: speculative decoding with it is ~2x faster output (setup.py's "
                      "words); it downloads ~5 GB of MTP tensors from the original checkpoint.")
                if auto_confirm:
                    print("  Not built under --yes: add --build-mtp to build it.")
                else:
                    want = safe_input("  Build the MTP draft layer now using tools/mtp_*.py? (y/N): ",
                                      default="n").lower() in ("y", "yes")
            if want:
                if not cls.prepare_mtp_draft_layer(m.mtp, strata_root):
                    return None
        if m.rt_complete():
            dv = m.rt / "draft_vocab.bin"
            if draft_vocab or not dv.exists():
                cls.refresh_draft_vocab(m.rt, strata_root, draft_vocab or "cjk")
            print(color_text(f" - MTP draft layer: {m.rt}", TermColor.GREEN))
            return m.rt
        return None

    @staticmethod
    def mtp_corrupt(mtp: Path, strata_root: Path) -> bool:
        """setup.py mtp_corrupt: tools/mtp_fetch.py verify exits 3 when fetched tensors are not the pinned ones."""
        if not (mtp / "tensors").is_dir():
            return False
        tool = StrataEnvironment.find_tool("mtp_fetch.py", strata_root)
        if not tool:
            return False
        r = subprocess.run([StrataEnvironment.find_strata_python(strata_root), str(tool), "verify", "--out", str(mtp)],
                           env=StrataEnvironment.tool_env(strata_root), stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        return r.returncode == 3

    @classmethod
    def refresh_draft_vocab(cls, rt_dir: Path, strata_root: Path, choice: str = "cjk") -> None:
        """setup.py refresh_draft_vocab: data/<subset> copied to rt/draft_vocab.bin when missing, or when another
        shipped subset is there; a subset made by hand is kept."""
        new, dst = strata_root / "data" / DRAFT_VOCABS.get(choice, "draft_vocab.bin"), rt_dir / "draft_vocab.bin"
        if not new.exists() or not rt_dir.is_dir():
            return
        if dst.exists():
            old = hashlib.sha256(dst.read_bytes()).hexdigest()
            shipped = OLD_DRAFT_VOCABS | {hashlib.sha256((strata_root / "data" / f).read_bytes()).hexdigest()
                                          for f in DRAFT_VOCABS.values() if (strata_root / "data" / f).exists()}
            if old not in shipped:
                print(f" - Draft vocabulary: a subset made by hand is in {dst}: kept")
                return
            if old == hashlib.sha256(new.read_bytes()).hexdigest():
                print(f" - Draft vocabulary ({choice}): {new.name} already in place")
                return
        shutil.copyfile(new, dst)
        print(f" - Draft vocabulary ({choice}): Copied {new.name} -> {dst}")

    @classmethod
    def prepare_mtp_draft_layer(cls, mtp_dir: Path, strata_root: Path) -> bool:
        """
        Builds the MTP speculative decoding draft layer in `mtp_dir` (the model's <artifacts>/<model>/mtp) using
        Strata's tools/mtp_fetch.py, tools/mtp_pack.py and tools/mtp_rt.py, with the same commands as Step 6 of
        setup.py.
        """
        rt_dir = mtp_dir / "rt"
        strata_py = StrataEnvironment.find_strata_python(strata_root)
        env = StrataEnvironment.tool_env(strata_root)
        fetch_tool = StrataEnvironment.find_tool("mtp_fetch.py", strata_root)
        pack_tool = StrataEnvironment.find_tool("mtp_pack.py", strata_root)
        rt_tool = StrataEnvironment.find_tool("mtp_rt.py", strata_root)

        if not (fetch_tool and pack_tool and rt_tool):
            print(color_text("Error: MTP tools (mtp_fetch.py, mtp_pack.py, mtp_rt.py) not found in Strata directory.", TermColor.RED))
            return False

        print(color_text("\n=== Building Strata MTP Draft Layer (Speculative Decoding) ===", TermColor.BOLD))
        packed_gguf = mtp_dir / "mtp-q2_0.gguf"
        steps = [
            ("[1/4] Fetching MTP draft tensors (~5 GB from original checkpoint)...",
             [strata_py, str(fetch_tool), "fetch", "--out", str(mtp_dir)]),
            ("[2/4] Packing MTP tensors with q2_0 experts...",
             [strata_py, str(pack_tool), "--src", str(mtp_dir), "--experts", "q2_0", "--out", str(packed_gguf)]),
            ("[3/4] Creating MTP runtime in rt/...",
             [strata_py, str(rt_tool), "--gguf", str(packed_gguf), "--out", str(rt_dir)]),
        ]
        for title, cmd in steps:
            print(color_text(title, TermColor.CYAN))
            rc, _ = run_streamed(cmd, env)
            if rc != 0:
                print(color_text(f"Error: {Path(cmd[1]).name} exited with code {rc}", TermColor.RED))
                return False

        print(color_text("[4/4] Verifying the fetched tensors (tools/mtp_fetch.py verify)...", TermColor.CYAN))
        if cls.mtp_corrupt(mtp_dir, strata_root):
            print(color_text("Error: MTP tensors reported corrupt or mismatched (mtp_fetch.py verify exit 3).",
                             TermColor.RED))
            return False
        print(color_text("MTP draft layer built and verified successfully!", TermColor.GREEN))
        return True


def print_bash_run_command(strata_path: Path, config_file: Path, port: int = 8080) -> None:
    """Displays copy-paste ready execution command with BASH syntax coloring."""
    strata_dir = str(strata_path.resolve())
    cfg_arg = config_arg(strata_path, config_file)

    is_tty = sys.stdout.isatty()
    RESET = "\033[0m" if is_tty else ""
    BOLD = "\033[1m" if is_tty else ""
    CYAN = "\033[96m" if is_tty else ""
    GREEN = "\033[92m" if is_tty else ""
    YELLOW = "\033[93m" if is_tty else ""
    WHITE = "\033[97m" if is_tty else ""

    border = f"{CYAN}{'═' * 76}{RESET}"

    print(f"\n{border}")
    print(f" {BOLD}{GREEN}✓ COMMAND TO RUN THE GENERATED MODEL (BASH){RESET}")
    print(f" {WHITE}Copy and paste into your terminal to start the Strata service:{RESET}\n")

    cmd1 = f"  {YELLOW}cd{RESET} {CYAN}{strata_dir}{RESET}"
    cmd2 = (
        f"  {GREEN}.venv/bin/python{RESET} {YELLOW}serve/server.py{RESET} "
        f"{CYAN}--engine{RESET} {WHITE}strata{RESET} "
        f"{CYAN}--config{RESET} {BOLD}{GREEN}{cfg_arg}{RESET} "
        f"{CYAN}--port{RESET} {WHITE}{port}{RESET}"
    )

    print(cmd1)
    print(cmd2)
    print()
    print(f" {CYAN}Single line command to copy and paste:{RESET}")
    print(f"  cd {strata_dir} && .venv/bin/python serve/server.py --engine strata --config {cfg_arg} --port {port}")
    print(f"{border}\n")


def config_arg(strata_path: Path, config_file: Path) -> str:
    """The --config value the server is started with from the Strata folder: the bare name when the JSON is in it."""
    config_file = Path(config_file).absolute()
    try:
        if config_file.parent.resolve() == strata_path.resolve():
            return config_file.name
    except OSError:
        pass
    return str(config_file)


# ==============================================================================
# 7. JSON CONFIGURATION GENERATION (--generate-model-file)
# ==============================================================================

class ConfigurationGenerator:
    """Builds and validates official Strata JSON configuration files."""

    @classmethod
    def generate_json_config(
        cls,
        model_or_pack_path: Path,
        output_json_path: Optional[Path] = None,
        context_tokens: int = DEFAULT_CONTEXT,
        kv_resident_tokens: Optional[int] = None,
        kv_streaming: str = "auto",
        vram_reserve_mib: Optional[int] = None,
        kv_quant: str = DEFAULT_KV_QUANT,
        vision: str = "auto",
        rope_scaling: Optional[str] = None,
        rope_scale: Optional[float] = None,
        low_ram: str = "off",
        adapt_async: int = 1,
        reasoning_budget: int = DEFAULT_REASONING_BUDGET,
        draft_vocab: Optional[str] = None,
        model_name: Optional[str] = None,
        port: int = 8080,
        strata_root: Optional[str] = None,
        strata_bin: Optional[str] = None,
        data_dir: Optional[str] = None,
        artifacts_dir: Optional[str] = None,
        artifacts_name: Optional[str] = None,
        build_mtp: bool = False,
        overwrite: bool = False,
        auto_confirm: bool = False,
        force: bool = False,
    ) -> Tuple[bool, Optional[Path]]:
        print(color_text("\n=== Hardware and Strata Environment Detection ===", TermColor.BOLD))
        hw = HardwareDetector.detect()
        strata_root_path = StrataEnvironment.find_strata_root(strata_root)
        data_dir_path = StrataEnvironment.find_strata_data_dir(data_dir, model_or_pack_path)
        model_info = ModelAnalyzer.inspect(model_or_pack_path, strata_root_path, data_dir_path)

        # Special handling if Strata-data root directory was provided
        if model_info.format == "STRATA_DATA_ROOT":
            print(color_text("\n[!] The specified path is the root Strata-data directory, not a model or pack.", TermColor.YELLOW))
            if model_info.available_packs and not auto_confirm:
                print(color_text("Available prepared packs in packs/:", TermColor.CYAN))
                for idx, p in enumerate(model_info.available_packs, 1):
                    print(f"  [{idx}] {p.name} ({p})")
                sel = safe_input("Select pack number to configure [1]: ", default="1")
                try:
                    pick_idx = int(sel) - 1
                    if not 0 <= pick_idx < len(model_info.available_packs):
                        raise ValueError
                except ValueError:
                    print(color_text("Invalid selection.", TermColor.RED))
                    return False, None
                model_or_pack_path = model_info.available_packs[pick_idx]
                model_info = ModelAnalyzer.inspect(model_or_pack_path, strata_root_path, data_dir_path)
            else:
                print(color_text("Name the model's GGUF or its pack folder with --model-dir.", TermColor.RED))
                for note in model_info.notes[1:]:
                    print(f" - {note}")
                return False, None

        if model_info.format == "GGUF_MULTI":
            pick = choose_candidate(model_info, auto_confirm)
            if pick is None:
                return False, None
            model_or_pack_path = pick
            model_info = ModelAnalyzer.inspect(pick, strata_root_path, data_dir_path)

        if not model_info.is_valid:
            print(color_text(f"Error: Specified path does not contain a recognizable or valid model: {model_or_pack_path}", TermColor.RED))
            for note in model_info.notes:
                print(f" - {note}")
            return False, None
        if model_info.format == "SAFETENSORS":
            print(color_text("Error: a SafeTensors model has no Strata pack; convert it to GGUF first.", TermColor.RED))
            return False, None

        # The model's own artifacts folder: <artifacts>/<model>/{packs, mtp, mtp/rt}, created when missing
        root = ModelArtifacts.root_for(data_dir_path, artifacts_dir)
        if root is None:
            print(color_text("Error: no Strata-data folder found; pass --artifacts-dir or --data-dir.", TermColor.RED))
            return False, None
        m = ModelArtifacts.for_model(model_info, root, artifacts_name)

        print(f"Processor: {hw.cpu_name} ({hw.cpu_cores} threads)")
        print(f"Total RAM: {hw.total_ram_gb:.2f} GB | Available: {hw.available_ram_gb:.2f} GB | Safe: {hw.safe_ram_gb:.2f} GB")
        if hw.gpus:
            for g in hw.gpus:
                print(f"GPU [{g['index']}]: {g['name']} ({g['type']})")
                print(f"   Total VRAM: {g['vram_total_mib']:.0f} MiB | Current Free VRAM: {g['vram_free_mib']:.0f} MiB")
        else:
            print(color_text("No compatible GPU detected.", TermColor.RED))

        print(f"Strata Root Directory: {strata_root_path or 'Not found'}")
        print(f"Data Directory: {data_dir_path or 'Not found'}")
        print(f"Model: {model_info.native_shard} (family: {model_info.family}, {model_info.num_experts or '?'} experts/layer)")
        print(f"Model artifacts folder: {m.base}")

        if not strata_root_path:
            print(color_text("Error: the Strata folder was not found (--strata-root).", TermColor.RED))
            return False, None
        own = m.owner()
        if own and not same_file(own, model_info.native_shard):
            print(color_text(f"Error: {m.base} belongs to {own}; choose another --artifacts-name.", TermColor.RED))
            return False, None
        mtp_rt_path = ArtifactGenerator.ensure_model_mtp(m, data_dir_path, strata_root_path, build_mtp,
                                                         draft_vocab, auto_confirm)
        if not mtp_rt_path:
            print(color_text("MTP Runtime: not built (no --mtp; '--build-mtp' builds it)", TermColor.YELLOW))

        try:
            scaling, scale = resolve_rope(context_tokens, rope_scaling, rope_scale)
        except ValueError as e:
            print(color_text(f"Error: {e}", TermColor.RED))
            return False, None

        plan = MemoryEstimator.evaluate(
            hw=hw,
            model=model_info,
            requested_context=context_tokens,
            preferred_kv_quant=kv_quant,
            kv_streaming=kv_streaming,
        )

        # --kv-resident: setup.py's rule unless given (0 = no KV streaming)
        recommended_kv_resident = plan.kv_resident_tokens
        chosen_kv_resident = kv_resident_tokens
        if chosen_kv_resident is None and not auto_confirm and context_tokens >= KV_STREAM_MIN_CTX:
            print(color_text("\n[i] KV streaming (--kv-resident):", TermColor.CYAN))
            print(f"    - Target context: {context_tokens:,} tokens (KV quantization: {kv_quant}, "
                  f"{plan.kv_bytes_per_token:,} bytes per token, {plan.kv_total_gb:.1f} GB in all)")
            print(f"    - Streamed, the whole KV cache lives in RAM ({plan.kv_stream_ram_gb:.1f} GB) and "
                  f"{KV_RESIDENT_TOKENS:,} positions per layer stay in VRAM; the VRAM it frees holds more experts")
            print("      (setup.py: +6% at 128K, +23% at 262K with Q2_0). The engine keeps at least "
                  f"{ENGINE_KV_RESIDENT_MIN:,}.")
            rec_txt = f"{recommended_kv_resident}" if recommended_kv_resident else "0 (off)"
            kv_in = safe_input(f"Enter 'kv-resident' [Recommended: {rec_txt}]: ", default=str(recommended_kv_resident))
            try:
                chosen_kv_resident = max(0, int(kv_in.strip()))
            except ValueError:
                print(color_text("Non-numeric input, using recommended size.", TermColor.YELLOW))
                chosen_kv_resident = recommended_kv_resident
        if chosen_kv_resident is None:
            chosen_kv_resident = recommended_kv_resident
        if chosen_kv_resident and chosen_kv_resident >= context_tokens:
            print(color_text(f"  [i] --kv-resident {chosen_kv_resident} holds the whole {context_tokens:,}-token "
                             "context: no streaming, flag left out.", TermColor.CYAN))
            chosen_kv_resident = 0
        if 0 < chosen_kv_resident < ENGINE_KV_RESIDENT_MIN:
            print(color_text(f"  [i] The engine raises --kv-resident {chosen_kv_resident} to its minimum "
                             f"{ENGINE_KV_RESIDENT_MIN}.", TermColor.YELLOW))
        plan.kv_resident_tokens = chosen_kv_resident

        print(color_text("\n=== Strata Resource Consumption Estimation ===", TermColor.BOLD))
        print(f"Requested context: {plan.requested_context:,} tokens")
        print(f"KV cache: {plan.kv_quant}, {plan.kv_total_gb:.2f} GB for the whole context")
        if plan.kv_resident_tokens:
            print(f"KV streaming: on, {plan.kv_resident_tokens:,} positions per layer in VRAM, "
                  f"{plan.kv_stream_ram_gb:.2f} GB in RAM")
        else:
            print("KV streaming: off (the KV cache is in VRAM)")
        print(f"Model in RAM (experts + ~{MODEL_RAM_BESIDE_EXPERTS_GB} GB): ~{plan.model_ram_gb:.1f} GB "
              f"of {plan.ram_total_gb:.0f} GB")
        print("VRAM: the engine sizes the expert cache to the free VRAM (--expert-cache auto), keeping "
              f"{vram_reserve_mib if vram_reserve_mib is not None else ENGINE_VRAM_RESERVE_MIB} MiB free")

        if plan.warning_messages:
            print(color_text("\nMemory warnings:", TermColor.YELLOW))
            for w in plan.warning_messages:
                print(f"  [!] {w}")
        for r in plan.recommendations:
            print(f"  [i] {r}")

        if not plan.is_viable and low_ram == "off":
            print(color_text("\n[!] Warning: Configuration exceeds estimated physical memory capacity.", TermColor.RED))
            if not force:
                if not auto_confirm:
                    confirm = safe_input("Do you want to force generation anyway? (y/N): ", default="n").lower()
                    if confirm not in ("y", "yes"):
                        print("Operation cancelled to prevent Out-Of-Memory (OOM).")
                        return False, None
                else:
                    print(color_text("Generation cancelled due to insufficient memory (use '--force' to bypass).", TermColor.RED))
                    return False, None

        # The pack: the model's own <artifacts>/<model>/packs (a ready-made pack: its linked copy there, else the
        # download's own pack/ folder)
        if model_info.format == "READY_PACK":
            pack_path = m.packs if (m.packs / "index.txt").exists() else model_info.ready["pack"]
        else:
            pack_path = m.packs
            same, how = pack_matches(pack_path, model_info.native_shard) if pack_complete(pack_path) else \
                (False, "missing or incomplete")
            if not same:
                print(color_text(f"\nCritical Error: this model's pack is not in {pack_path} ({how}).", TermColor.RED))
                print("To generate the required artifacts for Strata, run:")
                print(f"  ./strata_model_prep.py --convert-model --model-dir {model_info.native_shard}")
                return False, None

        native_path = model_info.native_shard
        ple_path = model_info.ple_shard
        if not native_path or not ple_path:
            print(color_text("\nCritical Error: Could not resolve shards for --native and --ple-gguf.", TermColor.RED))
            print(f" - native_path (--native):   {native_path or color_text('NOT FOUND', TermColor.RED)}")
            print(f" - ple_path (--ple-gguf):    {ple_path or color_text('NOT FOUND', TermColor.RED)}")
            return False, None

        # Resolve model configuration filename to match the input model filename exactly
        # e.g. Qwen3.8-Flash-Next-Uncensored.i1-IQ3_M.gguf -> Qwen3.8-Flash-Next-Uncensored.i1-IQ3_M.json
        if model_info.format == "READY_PACK":
            model_config_filename = f"{model_info.ready['dense'].parent.name}.json"
        else:
            model_config_filename = determine_model_config_filename(native_path)
        base_model_name = model_config_filename[:-5]
        model_display_name = base_model_name.lower()

        # Resolve destination path for JSON file (MODELNAME.json)
        default_strata_dir = strata_root_path.resolve()
        default_file_path = default_strata_dir / model_config_filename

        target_json_path: Path
        if output_json_path is not None:
            p_out = Path(output_json_path).expanduser()
            if p_out.is_dir():
                target_json_path = (p_out / model_config_filename).absolute()
            else:
                target_json_path = p_out.absolute()
        elif not auto_confirm:
            print(color_text("\n[i] Target Configuration File Destination (.json):", TermColor.CYAN))
            print(f"    Suggested model configuration filename: {color_text(model_config_filename, TermColor.GREEN)}")
            print(f"    Default directory (STRATA_PATH): {color_text(str(default_strata_dir), TermColor.GREEN)}")
            dest_in = safe_input_path(f"Enter target path or directory [{default_strata_dir}]: ", default=str(default_strata_dir))
            p_cand = Path(dest_in.strip() or str(default_strata_dir)).expanduser()
            if p_cand.is_dir():
                target_json_path = (p_cand / model_config_filename).absolute()
            elif p_cand.suffix.lower() == ".json":
                target_json_path = (default_strata_dir / p_cand.name) if len(p_cand.parts) == 1 else p_cand.absolute()
            else:
                p_cand.mkdir(parents=True, exist_ok=True)
                target_json_path = (p_cand / model_config_filename).absolute()
        else:
            target_json_path = default_file_path

        # An earlier config at this path: its API model name and log are kept (a calibration is saved under the model
        # name, setup.py hardware_key), and so are the keys added to it by hand (setup.py carry_over, below)
        old_cfg = None
        if target_json_path.is_file():
            try:
                old_cfg = json.loads(target_json_path.read_text(encoding="utf-8-sig"))
                if not isinstance(old_cfg, dict):
                    old_cfg = None
            except (OSError, ValueError):
                old_cfg = None
        if model_name:
            model_display_name = model_name
        elif old_cfg is not None and isinstance(old_cfg.get("model_name"), str) and old_cfg["model_name"]:
            model_display_name = old_cfg["model_name"]
        # setup.py: the log beside the configs, named as the config (strata-<tag>.json -> strata-<tag>.log)
        log_path = old_cfg.get("log") if old_cfg is not None and isinstance(old_cfg.get("log"), str) else \
            str(default_strata_dir / f"{target_json_path.stem.lower()}.log")

        # Expert profile: the one whose header has this model's experts per layer
        expert_profile, why_profile = StrataEnvironment.find_expert_profile(
            strata_root_path, model_info.num_experts, model_info.is_coder)
        if expert_profile is None:
            print(color_text(f"\nCritical Error: no expert profile for this model ({why_profile}).", TermColor.RED))
            return False, None

        engine_exe = Path(strata_bin).expanduser().absolute() if strata_bin else \
            StrataEnvironment.find_engine_binary(strata_root_path)
        if engine_exe is None or not engine_exe.is_file():
            print(color_text(f"\nCritical Error: the engine binary was not found ({engine_exe or 'engine/strata'}).",
                             TermColor.RED))
            return False, None
        cuda_lib_dirs = StrataEnvironment.find_cuda_lib_dirs(strata_root_path)

        # --- engine arguments, in setup.py step 7's order
        if model_info.format == "READY_PACK":
            r = model_info.ready
            model_args = ["--pack", str(pack_path), "--native", str(r["dense"]), "--native-dense-gguf", str(r["dense"]),
                          "--ple-gguf", str(r["ple"]), "--embd-gguf", str(r["embd"])]
        else:
            # a file of more than 2 shards: the engine finds the PLE table's shard itself from shard 1 (setup.py)
            model_args = ["--pack", str(pack_path.absolute()), "--native", str(native_path.absolute()),
                          *(["--ple-gguf", str(ple_path.absolute())] if len(model_info.gguf_shards) <= 2 else [])]
        engine_args: List[str] = [
            *model_args,
            "--expert-profile", str(expert_profile),
            "--expert-cache", "auto",
            "--prefill", DEFAULT_PREFILL_CHUNK,
            "--spec", DEFAULT_SPEC_DRAFTS,
            "--spec-min-p", DEFAULT_SPEC_MIN_P,
        ]
        if mtp_rt_path:
            engine_args += ["--mtp", str(mtp_rt_path)]
        engine_args += ["--max-context", str(plan.allocated_context)]
        if scaling is not None:
            engine_args += ["--rope-scaling", scaling, "--rope-scale", f"{scale:g}"]
            print(color_text(f" - RoPE scaling: {scaling} (factor: {scale:g}) [OK]", TermColor.CYAN))
        engine_args += ["--kv", plan.kv_quant]
        if plan.kv_resident_tokens:
            engine_args += ["--kv-resident", str(plan.kv_resident_tokens)]

        # The low-RAM mode (setup.py): the experts from the pack's experts.bin, copied into RAM or mapped
        if low_ram != "off" and model_info.format != "READY_PACK":
            if not (pack_path / "experts.bin").exists():
                print(color_text(f"\nCritical Error: --low-ram {low_ram} reads the pack's experts.bin, which "
                                 f"{pack_path} has not ('--convert-model --experts-bin' writes it).", TermColor.RED))
                return False, None
            engine_args.append("--resident-experts" if low_ram == "resident" else "--mmap-experts")
        if plan.unsloth_budget_gib is not None and low_ram == "off":
            engine_args += ["--resident-budget-gib", str(plan.unsloth_budget_gib)]

        # Images: the engine's --vision and the server's "vision" section (setup.py); there is no --mmproj option
        vision_section = None
        vision_mode = {"on": "gpu", "auto": "gpu"}.get(vision, vision)
        vexe = StrataEnvironment.find_engine_binary(strata_root_path, "strata-vision")
        if vision_mode in ("gpu", "cpu"):
            why_not = None
            if vexe is None:
                why_not = "the engine was built without images (no engine/strata-vision; BUILD.json \"vision\": " \
                          f"\"{StrataEnvironment.build_meta(strata_root_path).get('vision', '?')}\")"
            elif model_info.mmproj_file is None:
                why_not = "no mmproj-*.gguf vision encoder beside the model or in the models folder"
            if why_not:
                if vision in ("on", "gpu", "cpu"):
                    print(color_text(f"\nCritical Error: images asked for, but {why_not}.", TermColor.RED))
                    return False, None
                print(f" - Vision: off ({why_not})")
            else:
                engine_args += ["--vision", "--vram-reserve-mib", str(VISION_GPU_RESERVE_MIB)]
                vision_section = {"exe": str(vexe), "mmproj": str(model_info.mmproj_file),
                                  "model": str(native_path), "gpu": vision_mode == "gpu",
                                  "max_tokens": VISION_MAX_TOKENS[vision_mode]}
                if vision_mode == "cpu":
                    vision_section["threads"] = max(1, (os.cpu_count() or 8) // 2)
                print(f" - Vision encoder ({vision_mode}): {model_info.mmproj_file.name} [OK]")
        else:
            print(" - Vision: off")

        if vram_reserve_mib is not None:                # only when given (setup.py #493)
            if "--vram-reserve-mib" in engine_args:
                engine_args[engine_args.index("--vram-reserve-mib") + 1] = str(vram_reserve_mib)
            else:
                engine_args += ["--vram-reserve-mib", str(vram_reserve_mib)]

        if adapt_async:
            engine_args += ["--adapt-async", "1"]

        # Complete config structure compatible with serve/server.py
        tok_p = pack_path / "tokenizer"
        config_data: Dict[str, Any] = {
            "exe": str(engine_exe),
            "args": engine_args,
            "cwd": str(default_strata_dir),
            "tokenizer": str(tok_p.absolute()),
            "model_name": model_display_name,
            "log": log_path,
            "lib_dirs": cuda_lib_dirs,
            "port": port,
        }
        gpu = hw.primary_gpu
        if gpu is not None:
            config_data["gpu"] = gpu["index"]
            config_data["gpus_asked"] = True
        env: Dict[str, str] = {}
        if model_info.family == "unsloth":            # setup.py unsloth_default_env
            stem = gguf_base_name(native_path).upper()
            if "UD-Q5_K_XL" in stem or "UD-Q6_K_XL" in stem:
                env["STRATA_MMQ_KQ"] = "1"
            if "UD-" in stem and (gpu is None or gpu["type"] == "NVIDIA"):
                env["STRATA_HC_Q8"] = "1"
            if plan.unsloth_budget_gib is None:
                env["STRATA_ROUTE_TAIL_SKIP"] = "0"
        if gpu is not None and gpu["type"] == "AMD":
            config_data["backend"] = "hip"
        if env:                                       # setup.py writes "env" only when it has entries
            config_data["env"] = env
        if draft_vocab:
            config_data["draft_vocab"] = draft_vocab
        if vision_section:
            config_data["vision"] = vision_section
        if reasoning_budget:
            config_data["reasoning_budget_tokens"] = int(reasoning_budget)

        # Calibration measured for this PC and model (setup.py), else the hybrid-CPU pool-worker rule
        cal_date = apply_calibration(config_data, strata_root_path)
        if cal_date:
            print(color_text(f" - The settings tuned for this PC earlier are used ({cal_date})", TermColor.GREEN))
        else:
            config_data["args"], why_workers = recommend_pool_workers(config_data["args"], strata_root_path)
            print(f" - CPU expert workers (--pool-workers): {why_workers}")

        problems = engine_flag_problems(engine_exe, config_data["args"])
        if problems:
            print(color_text(f"\nCritical Error: the engine {engine_exe} does not know {', '.join(problems)} "
                             "(it would stop with 'unknown argument').", TermColor.RED))
            return False, None

        # Keep what the user added to an earlier config of this name (setup.py carry_over)
        kept: List[str] = []
        dropped: List[str] = []
        if old_cfg is not None:
            setup = load_setup(strata_root_path)
            if setup is not None:
                kept = setup.carry_over(old_cfg, config_data)
                dropped = setup.args_dropped(old_cfg, config_data)
            else:
                for k, v in old_cfg.items():
                    if k not in config_data:
                        config_data[k] = v
                        kept.append(k)

        json_output = json.dumps(config_data, indent=2)
        print(color_text("\n=== JSON Configuration for serve/server.py ===", TermColor.BOLD))
        print(json_output)
        if kept:
            print(color_text(f"  [i] kept from the earlier {target_json_path.name}: {', '.join(kept)}", TermColor.CYAN))
        if dropped:
            print(color_text(f"  [i] engine options the earlier {target_json_path.name} had that this one has not: "
                             f"{' '.join(dropped)}", TermColor.YELLOW))

        if old_cfg is not None and old_cfg == config_data:
            print(color_text(f"\n{target_json_path} is already this configuration: unchanged.", TermColor.GREEN))
        else:
            if target_json_path.exists() and not overwrite:
                print(color_text(f"\nNotice: File {target_json_path} already exists.", TermColor.YELLOW))
                if auto_confirm:
                    print(color_text("Pass --overwrite to replace it (a .bak copy is kept).", TermColor.RED))
                    return False, None
                confirm = safe_input("Do you want to overwrite it? (y/N): ", default="n").lower()
                if confirm not in ("y", "yes"):
                    print("Operation cancelled.")
                    return False, None

            if not auto_confirm:
                confirm = safe_input("\nDo you want to save this configuration? (Y/n): ", default="y").lower()
                if confirm in ("n", "no"):
                    print("Operation cancelled by user.")
                    return False, None

            # Backup existing configuration file (matches setup.py write_setup_config)
            if target_json_path.exists():
                bak_path = target_json_path.with_name(target_json_path.name + ".bak")
                try:
                    shutil.copyfile(target_json_path, bak_path)
                    print(color_text(f"  [i] Existing configuration backed up to: {bak_path.name}", TermColor.CYAN))
                except OSError as e:
                    print(color_text(f"  [!] could not keep a copy of the earlier {target_json_path.name} ({e})",
                                     TermColor.YELLOW))

            # Written whole or not at all (setup.py write_config)
            target_json_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = target_json_path.with_name(target_json_path.name + ".tmp")
            tmp.write_text(json_output + "\n", encoding="utf-8")
            os.replace(tmp, target_json_path)
            print(color_text(f"\nConfiguration successfully saved to: {target_json_path}!", TermColor.GREEN))

        # Write executable launch script run-<model>.sh (matches setup.py write_run_script)
        cls.write_run_script(strata_path=default_strata_dir, config_file=target_json_path, port=port)
        return True, target_json_path

    @classmethod
    def write_run_script(
        cls,
        strata_path: Path,
        config_file: Path,
        port: int = 8080,
    ) -> Optional[Path]:
        """Writes an executable run-<model>.sh (run-<model>.bat on Windows) launch script, as Step 7 of setup.py:
        the server with this config from the Strata folder."""
        model_tag = config_file.stem.lower()
        if model_tag.startswith("strata-"):           # setup.py: strata-<tag>.json is started by run-<tag>.sh
            model_tag = model_tag[len("strata-"):]
        cfg = config_arg(strata_path, config_file)
        # a launch script that already starts this config (setup.py's run-<tag>.sh, an earlier run of this script) is kept
        for existing in sorted(strata_path.glob("run-*.bat" if WIN else "run-*.sh")):
            try:
                text = existing.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if re.search(r'--config"?\s+"?(' + re.escape(str(config_file.absolute())) + "|" + re.escape(cfg) + r')"?(\s|$)',
                         text):
                print(color_text(f"  [✓] Launch script already starts this config: {existing}", TermColor.GREEN))
                return existing
        try:
            if WIN:
                script_path = strata_path / f"run-{model_tag}.bat"
                py = strata_path / ".venv" / "Scripts" / "python.exe"
                script_path.write_text(
                    "@echo off\r\ntitle Strata " + model_tag + "\r\ncd /d \"" + str(strata_path) + "\"\r\n"
                    f"\"{py if py.exists() else sys.executable}\" serve\\server.py --engine strata --config \"{cfg}\" "
                    f"--port {port}\r\nif errorlevel 1 pause\r\n", encoding="utf-8")
            else:
                script_path = strata_path / f"run-{model_tag}.sh"
                py = ".venv/bin/python" if (strata_path / ".venv" / "bin" / "python").exists() else sys.executable
                script_path.write_text(
                    "#!/bin/sh\n"
                    f'cd "{strata_path}"\n'
                    f'exec {py} serve/server.py --engine strata --config "{cfg}" --port {port}\n',
                    encoding="utf-8")
                script_path.chmod(0o755)
            print(color_text(f"  [✓] Executable launch script created: {script_path}", TermColor.GREEN))
            return script_path
        except OSError as e:
            print(color_text(f"  [!] the launch script could not be written ({e})", TermColor.YELLOW))
            return None


# ==============================================================================
# 8. SERVER VERIFICATION AND TESTING (--verify)
# ==============================================================================

class ServerValidator:
    """Checks a config the way serve/server.py and the engine read it, then test-starts the server."""

    @classmethod
    def verify_server_config(
        cls,
        config_path: Path,
        strata_root: Optional[Path] = None,
        test_port: int = 8099,
        timeout_s: float = 8.0,
        load_timeout_s: float = 0.0,
    ) -> bool:
        print(color_text("\n=== Verifying Effectiveness and Executability with Strata ===", TermColor.BOLD))
        config_path = config_path.expanduser().absolute()
        if not config_path.exists() and strata_root and (strata_root / config_path.name).exists():
            config_path = strata_root / config_path.name
        if not config_path.exists():
            print(color_text(f"Error: Configuration file not found: {config_path}", TermColor.RED))
            return False

        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))
        except ValueError as e:
            print(color_text(f"Error: {config_path} is not valid JSON: {e}", TermColor.RED))
            return False
        if not isinstance(cfg, dict) or not cfg.get("exe") or not isinstance(cfg.get("args"), list):
            print(color_text('Error: a run config is a JSON object with "exe" and an "args" list.', TermColor.RED))
            return False

        if not strata_root:
            cwd_str = cfg.get("cwd")
            strata_root = Path(cwd_str) if cwd_str else StrataEnvironment.find_strata_root()

        if not strata_root or not strata_root.exists():
            print(color_text(f"Error: Strata directory not found: {strata_root}", TermColor.RED))
            return False

        server_py = strata_root / "serve" / "server.py"
        strata_py = StrataEnvironment.find_strata_python(strata_root)
        fatal = []

        def ok_or(label: str, good: bool, bad_text: str, is_fatal: bool = True) -> None:
            print(f" - {label}: " + ("[OK]" if good else color_text(bad_text, TermColor.RED if is_fatal else TermColor.YELLOW)))
            if not good and is_fatal:
                fatal.append(label)

        # 1. Static check of referenced files and paths
        print("[1/2] Verifying existence of paths and artifacts in JSON:")
        args = cfg.get("args", [])
        val = lambda k: args[args.index(k) + 1] if k in args and args.index(k) + 1 < len(args) else None   # noqa: E731
        exe = Path(cfg["exe"]) if os.path.isabs(cfg["exe"]) else Path(cfg.get("cwd") or ".") / cfg["exe"]
        ok_or(f"Engine ({exe})", exe.is_file(), "[DOES NOT EXIST]")
        if exe.is_file():
            unknown = engine_flag_problems(exe, args)
            ok_or("Engine options", not unknown, f"[UNKNOWN TO THE ENGINE: {' '.join(unknown)} - exit code 2]")

        tokenizer_p = Path(cfg.get("tokenizer", ""))
        ok_or(f"Tokenizer ({tokenizer_p})", (tokenizer_p / "vocab.json").exists(), "[MISSING vocab.json]")

        pack = Path(val("--pack")) if val("--pack") else None
        if pack:
            ok_or(f"Strata Pack ({pack})", pack_complete(pack) or (pack / "index.txt").exists(),
                  "[MISSING index.txt / dense.bin / native_experts.txt]")
        else:
            ok_or("Strata Pack (--pack)", False, "[MISSING FROM ARGS]")

        native = Path(val("--native")) if val("--native") else None
        if native:
            ok_or(f"Primary Shard (--native) {native.name}", native.exists(), "[DOES NOT EXIST]")
            if pack and native.exists() and "--native-dense-gguf" not in args:
                same, how = pack_matches(pack, native)
                ok_or("Pack built from this --native", same, f"[NO: {how}]")
                shards, missing = expand_shards(native)
                ok_or("Shards beside --native", not missing, f"[MISSING {', '.join(missing)}]")
        else:
            ok_or("Primary Shard (--native)", False, "[MISSING FROM ARGS - FATAL]")
        if native and native.exists():
            files = expand_shards(native)[0] if "--native-dense-gguf" not in args else \
                [Path(v) for k, v in zip(args, args[1:]) if k in ("--native", "--ple-gguf", "--embd-gguf")]
            unmarked = [f.name for f in files if not done_mark(f).exists()]
            ok_or("Finish marks (.done) of the model files", not unmarked,
                  f"[none for {', '.join(unmarked)}: '--convert-model' writes them]", is_fatal=False)

        ple = Path(val("--ple-gguf")) if val("--ple-gguf") else None
        if ple:
            tensors = gguf_tensors(ple, strata_root)
            has = tensors is None or PLE_TENSOR in tensors
            ok_or(f"PLE Shard (--ple-gguf) {ple.name}", ple.exists() and has,
                  "[DOES NOT EXIST]" if not ple.exists() else f"[NO {PLE_TENSOR} IN IT]")
        elif native and native.exists():
            shards, _ = expand_shards(native)
            print(" - PLE Shard (--ple-gguf): not given; "
                  + ("the engine finds it from shard 1 (more than 2 shards, as setup.py writes it)" if len(shards) > 2
                     else color_text("[setup.py gives it for files of 1 or 2 shards]", TermColor.YELLOW)))

        prof = Path(val("--expert-profile")) if val("--expert-profile") else None
        if prof:
            shape = profile_shape(prof)
            ok_or(f"Expert profile ({prof.name})", shape is not None, "[MISSING OR NOT A PROFILE]")
            if shape and native and native.exists():
                meta = GGUFReader.read_metadata(native)
                ne = meta.get(f"{meta.get('general.architecture', '')}.expert_count")
                if ne:
                    ok_or(f"Profile experts per layer ({shape[1]}) = model's ({ne})", shape[1] == int(ne),
                          "[MISMATCH: use the profile of this model's expert count]")

        mtp = Path(val("--mtp")) if val("--mtp") else None
        if mtp:
            ok_or(f"MTP Runtime ({mtp})", (mtp / "experts.bin").exists() and (mtp / "dense.bin").exists(),
                  "[MISSING experts.bin / dense.bin]")
            ok_or("MTP draft vocabulary (draft_vocab.bin)", (mtp / "draft_vocab.bin").exists(),
                  "[MISSING: the draft head runs without its subset]", is_fatal=False)
        else:
            print(color_text(" - MTP Runtime (--mtp): not set (no MTP drafting)", TermColor.YELLOW))
        if pack and mtp:
            isolated = pack.name == "packs" and os.path.abspath(str(mtp)) == os.path.abspath(str(pack.parent / "mtp" / "rt"))
            print(" - Artifacts layout: " + ("[OK] one folder per model: " + str(pack.parent) + "/{packs, mtp/rt}"
                                            if isolated else color_text("[pack and MTP are not in one "
                                                                        "artifacts/<model> folder]", TermColor.YELLOW)))

        if "--pool-workers" in args:
            print(f" - Expert Pool Workers (--pool-workers): {val('--pool-workers')} [OK]")
        kvr = val("--kv-resident")
        if kvr:
            ok_or(f"KV streaming (--kv-resident {kvr})", kvr.isdigit(), "[NOT A NUMBER]")

        lib_dirs = cfg.get("lib_dirs", [])
        missing_libs = [d for d in lib_dirs if not Path(d).is_dir()]
        ok_or(f"CUDA Directories (lib_dirs) {lib_dirs}", not missing_libs, f"[MISSING {missing_libs}]",
              is_fatal=False)
        print(f" - GPU Allocation: {cfg.get('gpu', 'engine default')} (gpus_asked: {cfg.get('gpus_asked')})")

        vis = cfg.get("vision")
        if isinstance(vis, dict):
            for k in ("exe", "mmproj", "model"):
                p = Path(vis.get(k, ""))
                ok_or(f"Vision {k} ({p.name})", p.is_file(), "[NOT FOUND]")
            ok_or("Vision flag (--vision)", "--vision" in args, "[the server adds it]", is_fatal=False)
        elif "--vision" in args:
            ok_or("Vision section", False, '[--vision without a "vision" section: no encoder runs]', is_fatal=False)
        else:
            print(" - Vision: [DISABLED - Text-only mode]")

        if fatal:
            print(color_text(f"\n[FAILED] {len(fatal)} problem(s) would stop the engine at start (EngineDied).",
                             TermColor.RED))
            return False

        # 2. Startup test with serve/server.py: --lazy (config, tokenizer, port) or a real load
        lazy = load_timeout_s <= 0
        print(f"\n[2/2] Testing serve/server.py startup on test port {test_port} "
              + ("(--lazy: the server, not the engine)..." if lazy else f"(loading the model, up to {load_timeout_s:.0f} s)..."))
        cmd = [strata_py, str(server_py), "--engine", "strata", "--config", str(config_path), "--port", str(test_port)]
        if lazy:
            cmd.append("--lazy")
        wait_s = timeout_s if lazy else load_timeout_s

        proc = None
        try:
            proc = subprocess.Popen(cmd, cwd=str(strata_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True)
            start_time = time.monotonic()
            while time.monotonic() - start_time < wait_s:
                if proc.poll() is not None:
                    out, _ = proc.communicate()
                    print(color_text(f"Server exited unexpectedly (Code: {proc.returncode}):", TermColor.RED))
                    print(out[-4000:])
                    log = cfg.get("log")
                    if log and Path(log).is_file():
                        print(color_text(f"Last lines of {log}:", TermColor.YELLOW))
                        print("".join(Path(log).read_text(encoding="utf-8", errors="replace").splitlines(True)[-25:]))
                    return False
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{test_port}/v1/models", timeout=1.0) as resp:
                        if resp.status == 200:
                            data = json.loads(resp.read().decode("utf-8"))
                            models = [m.get("id") for m in data.get("data", [])]
                            print(color_text(" Server ready and responding successfully!", TermColor.GREEN))
                            print(f" Models registered in API: {models}")
                            if lazy:
                                print(color_text("\n[SUCCESS] Config, tokenizer and engine options check out; the "
                                                 "engine itself loads with the first request (--verify-load starts it "
                                                 "now).", TermColor.GREEN))
                            else:
                                print(color_text("\n[SUCCESS] The engine loaded the model with this configuration.",
                                                 TermColor.GREEN))
                            return True
                except Exception:
                    time.sleep(0.5 if lazy else 2.0)

            print(color_text(f"Warning: Server took longer than {wait_s:.0f}s to initialize on port {test_port}.", TermColor.YELLOW))
            return False

        finally:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10.0)
                except Exception:
                    proc.kill()


# ==============================================================================
# 9. PERFORMANCE OPTIMIZATION SUGGESTIONS (--suggestions)
# ==============================================================================

class PerformanceAdvisor:
    """Settings this script writes or offers, with what Strata's own sources say about them."""

    @classmethod
    def display_suggestions(cls) -> None:
        is_tty = sys.stdout.isatty()
        RESET = "\033[0m" if is_tty else ""
        BOLD = "\033[1m" if is_tty else ""
        CYAN = "\033[96m" if is_tty else ""
        GREEN = "\033[92m" if is_tty else ""
        YELLOW = "\033[93m" if is_tty else ""
        WHITE = "\033[97m" if is_tty else ""

        border = f"{CYAN}{'═' * 78}{RESET}"
        divider = f"{CYAN}{'─' * 78}{RESET}"

        print(f"\n{border}")
        print(f" {BOLD}{GREEN}★ STRATA PERFORMANCE TUNING & OPTIMIZATION SUGGESTIONS ★{RESET}")
        print(f" {WHITE}What each setting does, from Strata's setup.py, engine and docs:{RESET}")
        print(f"{border}\n")

        suggestions = [
            (
                "--kv-quant {int8, q4_0, k8v4, fp16}  (engine: --kv)",
                "int8 (setup.py's default past 8K) | q4_0 for the longest contexts",
                [
                    ("Bytes", "Per context token, 13 attention layers: int8 13.7 KB, k8v4 10.6 KB, q4_0 7.5 KB, "
                              "fp16 27.5 KB (setup.py). 128K with int8: 1.8 GB."),
                ],
            ),
            (
                "--compat-bf16 {auto,on,off}  (iq_pack.py --compat-bf16)",
                "auto",
                [
                    ("Details", "setup.py passes --compat-bf16 for Unsloth's files only. It dequantizes the small "
                                "projections the engine reads as BF16 (rounding to nearest-even); experts and the PLE "
                                "table stay unchanged. auto: Unsloth files, or any file iq_pack.py refuses without it."),
                ],
            ),
            (
                "--adapt-async 1",
                "1 (written by default; the engine turns it off itself where it cannot run)",
                [
                    ("Measured", "docs/EXCHANGE_ROTATION.md, 23 coding-agent turns: 166.0 / 170.1 tok/s and "
                                 "185.6 / 182.5 tok/s with exchange rotation and --adapt-async 1 against the "
                                 "blocking tier's 147.8-159.2 / 172.8-184.9."),
                ],
            ),
            (
                "--vision {auto,gpu,cpu,off}",
                "off for text-only work",
                [
                    ("Details", "Images need engine/strata-vision and an mmproj-*.gguf; the config gets --vision and a "
                                "\"vision\" section (the engine has no --mmproj option). The GPU encoder takes ~1.2 GB "
                                "at 1024 image tokens (setup.py)."),
                ],
            ),
            (
                "--kv-resident <tokens>  (KV streaming)",
                "32768 from a 64K context up, when the RAM holds the KV cache (setup.py)",
                [
                    ("Measured", "setup.py: the VRAM it frees holds more experts, +6% at 128K and +23% at 262K "
                                 "with Q2_0. The engine keeps at least 20480 positions per layer."),
                ],
            ),
            (
                "--pool-workers <N>",
                "hybrid CPU with more E- than P-cores: P - 1 + E // 2; else the engine's own count",
                [
                    ("Measured", "setup.py #642, i9-14900KF (8P + 16E): 15 workers decoded 165 / 116 tok/s against "
                                 "106 / 84 with all 23. A calibration (./setup.sh --calibrate) wins over the rule."),
                ],
            ),
            (
                "--expert-profile <file> & --expert-cache auto",
                "data/expert-profile.bin (512 experts) or expert-profile-coder.bin (256, the Coder)",
                [
                    ("Details", "The profile must have the model's experts per layer (its header says it); the "
                                "engine sizes the expert cache to the free VRAM."),
                ],
            ),
            (
                "--spec 4 & --spec-min-p 0.5 (with --mtp <path>)",
                "--spec 4 --spec-min-p 0.5 (setup.py)",
                [
                    ("Details", "The MTP draft layer drafts tokens the model verifies; setup.py: ~2x faster output. "
                                "--draft-vocab en needs ~133 MiB of VRAM for the draft head against cjk's ~348."),
                ],
            ),
            (
                "--prefill auto",
                "auto",
                [
                    ("Details", "The engine picks the prompt chunk itself (up to 8192, as the free VRAM allows); "
                                "setup.py: about 2x faster on long prompts than the old fixed 2048."),
                ],
            ),
        ]

        for idx, (param, rec, points) in enumerate(suggestions, 1):
            print(f"{BOLD}{CYAN}{idx}. {param}{RESET}")
            print(f"   {YELLOW}• Recommended:{RESET} {WHITE}{rec}{RESET}")
            print(f"   {GREEN}• Benefit / Details:{RESET}")
            for sub_title, sub_desc in points:
                if sub_title in ("Benefit", "Details"):
                    print(f"     {sub_desc}")
                else:
                    print(f"     - {BOLD}{sub_title}{RESET}: {sub_desc}")
            if idx < len(suggestions):
                print(f"{divider}")

        print(f"\n{border}\n")


# ==============================================================================
# 10. COMMAND-LINE INTERFACE (CLI)
# ==============================================================================

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strata_model_prep.py",
        description="Model preparation and configuration automation for the Strata Inference Engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
---------------
1. Build a downloaded GGUF's artifacts (shard 1, or the single file) into <data>/artifacts/<model>/{packs,mtp/rt}:
   ./strata_model_prep.py --convert-model \\
       --model-dir /home/USERNAME/Documents/AI/llm_models/Strata-data/models/mradermacher_Qwen3.8-Flash-Next-Uncensored-i1-GGUF/Qwen3.8-Flash-Next-Uncensored.i1-IQ3_M.gguf

2. Generate JSON configuration file for serve/server.py (from the GGUF or its artifacts/<model>/packs):
   ./strata_model_prep.py --generate-model-file \\
       --model-dir /home/USERNAME/Documents/AI/llm_models/Strata-data/artifacts/qwen3.8-flash-next-uncensored.i1-iq3_m/packs \\
       --context 131072

3. Complete workflow: pack, build the MTP draft layer, write the config and verify it:
   ./strata_model_prep.py --convert-model --build-mtp --generate-model-file --verify \\
       --model-dir /home/USERNAME/Documents/AI/llm_models/Strata-data/models/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF__swift-IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \\
       --context 131072 --yes

4. Display recommended performance tuning settings and explanations:
   ./strata_model_prep.py --suggestions

5. Verify an existing JSON configuration (add --verify-load to load the model too):
   ./strata_model_prep.py --verify --config-out Qwen3.8-Flash-Next-Uncensored.i1-IQ3_M.json
""",
    )

    mode_group = parser.add_argument_group("Operation Modes")
    mode_group.add_argument(
        "--convert-model",
        action="store_true",
        help="Prepare downloaded model using Strata's tools/iq_pack.py and generate required artifacts.",
    )
    mode_group.add_argument(
        "--generate-model-file",
        action="store_true",
        help="Generate optimized JSON configuration file for serve/server.py.",
    )
    mode_group.add_argument(
        "--verify",
        action="store_true",
        help="Check the config against the engine and test serve/server.py startup on a test port.",
    )
    mode_group.add_argument(
        "--verify-load",
        type=float,
        nargs="?",
        const=900.0,
        default=0.0,
        metavar="SECONDS",
        help="With --verify: start the engine for real and wait until the model is loaded (default 900 s).",
    )
    mode_group.add_argument(
        "--suggestions",
        action="store_true",
        help="Display recommended performance tuning settings and explanations of their benefits.",
    )

    path_group = parser.add_argument_group("Paths and Files")
    path_group.add_argument(
        "--model-dir",
        type=str,
        help="Downloaded model: a GGUF file (any shard), its folder, an existing Strata pack, or a ready-made pack folder.",
    )
    path_group.add_argument(
        "--config-out",
        type=str,
        default=None,
        help="Path or directory for the output JSON file (defaults to '<MODELNAME>.json' in STRATA_PATH).",
    )
    path_group.add_argument(
        "--artifacts-dir", "--output-dir",
        dest="artifacts_dir",
        type=str,
        default=None,
        help="The artifacts folder (created when missing; default: <data>/artifacts). Each model gets its own "
             "<folder>/<model>/packs, <model>/mtp and <model>/mtp/rt.",
    )
    path_group.add_argument(
        "--artifacts-name",
        type=str,
        default=None,
        help="The model's folder name inside the artifacts folder (default: the GGUF's name in lower case; "
             "<model folder>__<name> when that one belongs to another GGUF of the same name).",
    )
    path_group.add_argument(
        "--strata-root",
        type=str,
        default=None,
        help="Path to Strata software directory (default: this script's folder, else /home/USERNAME/Documents/AI/bin/Strata).",
    )
    path_group.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Strata-data folder (models/, packs/, mtp/); default: the one the model is in, else "
             f"{DEFAULT_DATA_DIR}.",
    )
    path_group.add_argument(
        "--strata-bin",
        type=str,
        help="Path to engine executable binary (default: engine/strata, else build/strata).",
    )

    param_group = parser.add_argument_group("Inference Parameters and Resources")
    param_group.add_argument(
        "--context",
        type=int,
        help="Context window size in tokens (e.g. 65536, 131072, 262144, 524288).",
    )
    param_group.add_argument(
        "--rope-scaling",
        choices=["yarn", "linear", "none"],
        default=None,
        help="RoPE scaling method (past 262144 tokens: yarn by default; 'none' is refused there).",
    )
    param_group.add_argument(
        "--rope-scale",
        type=float,
        default=None,
        help="Explicit RoPE scale factor (defaults to context / 262144, at least 1).",
    )
    param_group.add_argument(
        "--kv-resident",
        type=int,
        default=None,
        help="KV streaming: positions per layer kept in VRAM (default: setup.py's rule, 32768 from 64K when the "
             "RAM holds the KV cache; 0 = off).",
    )
    param_group.add_argument(
        "--kv-streaming",
        choices=["auto", "on", "off"],
        default="auto",
        help="setup.py --kv-streaming: on/off overrides the RAM test (default: auto).",
    )
    param_group.add_argument(
        "--kv-quant",
        choices=SUPPORTED_KV_QUANTS,
        default=DEFAULT_KV_QUANT,
        help=f"KV cache quantization, the engine's --kv (default: {DEFAULT_KV_QUANT}).",
    )
    param_group.add_argument(
        "--vision",
        choices=["auto", "on", "gpu", "cpu", "off"],
        default="auto",
        help="Images: auto (on when engine/strata-vision and an mmproj file exist), on/gpu, cpu, or off.",
    )
    param_group.add_argument(
        "--draft-vocab",
        choices=list(DRAFT_VOCABS),
        default=None,
        help="Draft vocabulary subset for MTP speculative decoding (default: cjk; en uses the least VRAM).",
    )
    param_group.add_argument(
        "--vram-reserve",
        type=int,
        default=None,
        help=f"VRAM in MiB kept free for other programs (--vram-reserve-mib; engine default {ENGINE_VRAM_RESERVE_MIB}).",
    )
    param_group.add_argument(
        "--low-ram",
        choices=["off", "mmap", "resident"],
        default="off",
        help="setup.py's low-RAM mode: the experts from the pack's experts.bin, mapped or copied into RAM "
             "(needs --convert-model --experts-bin).",
    )
    param_group.add_argument(
        "--adapt-async",
        type=int,
        choices=[0, 1],
        default=1,
        help="Write --adapt-async 1 (default 1; the engine turns it off itself where it cannot run).",
    )
    param_group.add_argument(
        "--reasoning-budget",
        type=int,
        default=DEFAULT_REASONING_BUDGET,
        help=f"reasoning_budget_tokens in the config (default {DEFAULT_REASONING_BUDGET}; 0 = no budget, key left out).",
    )
    param_group.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="\"model_name\" in the config (the API's model id; default: the earlier config's, else the GGUF's name).",
    )
    param_group.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port written into the config and the launch script (default 8080).",
    )

    exec_group = parser.add_argument_group("Packing and Execution Options")
    exec_group.add_argument(
        "--build-mtp",
        action="store_true",
        help="Build MTP speculative decoding draft layer runtime (tools/mtp_fetch.py, mtp_pack.py, mtp_rt.py) if missing.",
    )
    exec_group.add_argument(
        "--compat-bf16",
        choices=["auto", "on", "off"],
        default="auto",
        help="tools/iq_pack.py --compat-bf16: auto = Unsloth files, or when iq_pack.py asks for it (default).",
    )
    exec_group.add_argument(
        "--no-compat-bf16",
        action="store_true",
        help="Same as --compat-bf16 off.",
    )
    exec_group.add_argument(
        "--experts-bin",
        action="store_true",
        help="Generate 'experts.bin' when packing (the low-RAM mode's file; otherwise Strata reads the GGUFs).",
    )
    exec_group.add_argument(
        "--force",
        action="store_true",
        help="Force configuration generation even if estimated physical memory is exceeded.",
    )
    exec_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files and directories without prompt confirmation.",
    )
    exec_group.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Accept interactive confirmations automatically (non-interactive / CI mode).",
    )

    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    # Display suggestions if requested
    if args.suggestions:
        PerformanceAdvisor.display_suggestions()
        if not args.convert_model and not args.generate_model_file and not args.verify:
            return 0

    if not args.convert_model and not args.generate_model_file and not args.verify:
        parser.print_help()
        print(color_text("\nYou must specify at least one action: '--convert-model', '--generate-model-file', '--verify', or '--suggestions'.", TermColor.YELLOW))
        return 1

    pack_output_path: Optional[Path] = None
    configured_json_path: Optional[Path] = None
    verified_json_path: Optional[Path] = None
    strata_root_path = StrataEnvironment.find_strata_root(args.strata_root) or DEFAULT_STRATA_ROOT
    compat = "off" if args.no_compat_bf16 else args.compat_bf16

    # 1. Execute --convert-model
    if args.convert_model:
        print(color_text("=== Model Conversion Mode (--convert-model) ===", TermColor.BOLD))

        model_dir_str = args.model_dir
        if not model_dir_str:
            model_dir_str = safe_input_path("Enter downloaded model directory or file path: ")
            if not model_dir_str:
                print(color_text("Error: Source model path not specified.", TermColor.RED))
                return 1

        if not args.artifacts_dir and not args.yes:
            args.artifacts_dir = safe_input_path(
                "Enter the artifacts folder (each model gets <folder>/<model>/packs, mtp, mtp/rt) "
                "[<data>/artifacts]: ") or None

        success, final_model_path = ArtifactGenerator.generate_artifacts(
            source_model_dir=Path(model_dir_str).expanduser(),
            compat_bf16=compat,
            experts_bin=args.experts_bin,
            build_mtp=args.build_mtp,
            draft_vocab=args.draft_vocab,
            overwrite=args.overwrite,
            auto_confirm=args.yes,
            strata_root=args.strata_root,
            data_dir=args.data_dir,
            artifacts_dir=Path(args.artifacts_dir) if args.artifacts_dir else None,
            artifacts_name=args.artifacts_name,
        )
        if not success or not final_model_path:
            print(color_text("Model conversion did not complete successfully.", TermColor.RED))
            return 2

        pack_output_path = final_model_path

    # 2. Execute --generate-model-file
    if args.generate_model_file:
        print(color_text("\n=== JSON Configuration Generation Mode (--generate-model-file) ===", TermColor.BOLD))

        config_model_path_str = args.model_dir
        if pack_output_path:
            config_model_path_str = str(pack_output_path)
        elif not config_model_path_str:
            config_model_path_str = safe_input_path("Enter Strata model or pack directory path: ")
            if not config_model_path_str:
                print(color_text("Error: Model path not specified.", TermColor.RED))
                return 1

        ctx = args.context
        if ctx is None:
            if not args.yes:
                print(color_text("\n[i] Select Context Window (Tokens) [aligned with setup.py]:", TermColor.CYAN))
                for idx, c in enumerate(STANDARD_CONTEXTS, 1):
                    tag_c = ""
                    if c == DEFAULT_CONTEXT:
                        tag_c = f" {TermColor.GREEN}[Recommended default]{TermColor.RESET}"
                    elif c == 65536:
                        tag_c = f" {TermColor.CYAN}[KV streaming from here up]{TermColor.RESET}"
                    elif c == 204800:
                        tag_c = " [inside the trained 262144: no rope scaling]"
                    elif c == TRAINED_CONTEXT_LIMIT:
                        tag_c = " [Max native trained context]"
                    elif c > TRAINED_CONTEXT_LIMIT:
                        tag_c = f" {TermColor.YELLOW}[RoPE YaRN extension]{TermColor.RESET}"
                    print(f"  {idx}) {c // 1024}K tokens ({c:,}){tag_c}")

                default_pick = str(STANDARD_CONTEXTS.index(DEFAULT_CONTEXT) + 1)
                ctx_input = safe_input(f"Choose context option [1-{len(STANDARD_CONTEXTS)}] or enter tokens directly "
                                       f"[default: {default_pick} (128K)]: ", default=default_pick).strip()
                if ctx_input.isdigit():
                    pick = int(ctx_input)
                    ctx = STANDARD_CONTEXTS[pick - 1] if 1 <= pick <= len(STANDARD_CONTEXTS) else pick
                else:
                    ctx = DEFAULT_CONTEXT
            else:
                ctx = DEFAULT_CONTEXT
        if ctx < 1024:
            print(color_text(f"Error: a context of {ctx} tokens is too small.", TermColor.RED))
            return 1

        out_json = Path(args.config_out) if args.config_out else None

        success, configured_json_path = ConfigurationGenerator.generate_json_config(
            model_or_pack_path=Path(config_model_path_str).expanduser(),
            output_json_path=out_json,
            context_tokens=ctx,
            kv_resident_tokens=args.kv_resident,
            kv_streaming=args.kv_streaming,
            vram_reserve_mib=args.vram_reserve,
            kv_quant=args.kv_quant,
            vision=args.vision,
            rope_scaling=args.rope_scaling,
            rope_scale=args.rope_scale,
            low_ram=args.low_ram,
            adapt_async=args.adapt_async,
            reasoning_budget=args.reasoning_budget,
            draft_vocab=args.draft_vocab,
            model_name=args.model_name,
            port=args.port,
            strata_root=args.strata_root,
            strata_bin=args.strata_bin,
            data_dir=args.data_dir,
            artifacts_dir=args.artifacts_dir,
            artifacts_name=args.artifacts_name,
            build_mtp=args.build_mtp,
            overwrite=args.overwrite,
            auto_confirm=args.yes,
            force=args.force,
        )
        if not success:
            return 3

    # 3. Execute --verify
    if args.verify:
        config_path = (
            configured_json_path
            if (args.generate_model_file and configured_json_path)
            else (Path(args.config_out) if args.config_out else None)
        )
        if not config_path:
            config_input = safe_input_path("Enter JSON configuration file path to verify: ")
            if not config_input:
                print(color_text("Error: JSON configuration file path not specified for verification.", TermColor.RED))
                return 1
            config_path = Path(config_input)

        success = ServerValidator.verify_server_config(
            config_path=config_path,
            strata_root=strata_root_path,
            test_port=8099,
            timeout_s=8.0,
            load_timeout_s=args.verify_load,
        )
        if not success:
            return 4
        verified_json_path = config_path

    # Final step: Display copy-paste ready BASH command
    final_json = configured_json_path or verified_json_path or (Path(args.config_out) if args.config_out else None)
    if final_json:
        print_bash_run_command(strata_root_path, final_json, port=args.port)

    return 0


if __name__ == "__main__":
    sys.exit(main())
