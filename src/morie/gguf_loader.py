"""GGUF model file loader for MORIE's inference engine.

Parses GGUF v3 files (the format used by Ollama and llama.cpp) and
provides access to model metadata and weight tensors.

References
----------
* GGUF spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
* ligguf (reference): https://github.com/matrixsmaster/ligguf
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# GGUF magic number: "GGUF" in little-endian
GGUF_MAGIC = 0x46554747  # 'G','G','U','F'

# GGML tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_IQ4_NL = 20
GGML_TYPE_TQ2 = 100
GGML_TYPE_TQ3 = 101
GGML_TYPE_TQ4 = 102

# Bytes per element for each type (approximate for block-quantized types)
_TYPE_SIZES = {
    GGML_TYPE_F32: 4.0,
    GGML_TYPE_F16: 2.0,
    GGML_TYPE_Q4_0: 0.5 + 2 / 32,  # 18 bytes per 32 elements
    GGML_TYPE_Q4_K: 0.5625,  # 144 bytes per 256 elements
    GGML_TYPE_Q8_0: 1.0 + 4 / 32,  # 36 bytes per 32 elements
    GGML_TYPE_TQ2: (4 + 2 + 64) / 256,  # 6 + 64 bytes per 256 elements
    GGML_TYPE_TQ3: (4 + 2 + 96) / 256,  # 6 + 96 bytes per 256 elements
    GGML_TYPE_TQ4: (4 + 2 + 128) / 256,  # 6 + 128 bytes per 256 elements
}

# Block sizes for quantized types
_BLOCK_SIZES = {
    GGML_TYPE_Q4_0: 32,
    GGML_TYPE_Q4_K: 256,
    GGML_TYPE_Q8_0: 32,
    GGML_TYPE_Q5_K: 256,
    GGML_TYPE_Q6_K: 256,
}

# GGUF value types
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12


@dataclass
class TensorInfo:
    """Metadata for one tensor in the GGUF file."""

    name: str
    shape: tuple[int, ...]
    dtype: int  # GGML type ID
    offset: int  # byte offset from start of tensor data section
    n_elements: int = 0

    @property
    def dtype_name(self) -> str:
        names = {0: "F32", 1: "F16", 2: "Q4_0", 8: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K"}
        return names.get(self.dtype, f"type_{self.dtype}")


class GGUFModel:
    """Load and parse a GGUF model file.

    Parameters
    ----------
    path : str or Path
        Path to the GGUF file.

    Examples
    --------
    >>> model = GGUFModel("~/.ollama/models/blobs/sha256-abc123")
    >>> print(model.config)
    {'architecture': 'llama', 'n_layers': 32, ...}
    >>> names = model.tensor_names()
    >>> weight = model.get_tensor("token_embd.weight")
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self._metadata: dict[str, Any] = {}
        self._tensors: dict[str, TensorInfo] = {}
        self._data_offset: int = 0
        self._mm: mmap.mmap | None = None
        self._fp = None

        if not self.path.exists():
            raise FileNotFoundError(f"GGUF file not found: {self.path}")

        self._fp = open(self.path, "rb")

        self._parse_header()

    def _parse_header(self) -> None:
        """Parse the GGUF header, metadata, and tensor info."""
        fp = self._fp

        # Magic
        magic = struct.unpack("<I", fp.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic: 0x{magic:08X}, expected 0x{GGUF_MAGIC:08X})")

        # Version
        version = struct.unpack("<I", fp.read(4))[0]
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")

        # Counts
        n_tensors = struct.unpack("<Q", fp.read(8))[0]
        n_kv = struct.unpack("<Q", fp.read(8))[0]

        # Metadata KV pairs
        for _ in range(n_kv):
            key = self._read_string(fp)
            val = self._read_value(fp)
            self._metadata[key] = val

        # Tensor info
        for _ in range(n_tensors):
            name = self._read_string(fp)
            n_dims = struct.unpack("<I", fp.read(4))[0]
            shape = tuple(struct.unpack("<Q", fp.read(8))[0] for _ in range(n_dims))
            dtype = struct.unpack("<I", fp.read(4))[0]
            offset = struct.unpack("<Q", fp.read(8))[0]
            n_elem = 1
            for s in shape:
                n_elem *= s
            self._tensors[name] = TensorInfo(
                name=name,
                shape=shape,
                dtype=dtype,
                offset=offset,
                n_elements=n_elem,
            )

        # Align to 32 bytes for tensor data
        pos = fp.tell()
        alignment = self._metadata.get("general.alignment", 32)
        pad = (alignment - (pos % alignment)) % alignment
        self._data_offset = pos + pad

        # mmap the file for efficient tensor access
        self._mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)

    @staticmethod
    def _read_string(fp) -> str:
        length = struct.unpack("<Q", fp.read(8))[0]
        return fp.read(length).decode("utf-8")

    def _read_value(self, fp) -> Any:
        vtype = struct.unpack("<I", fp.read(4))[0]
        return self._read_typed_value(fp, vtype)

    def _read_typed_value(self, fp, vtype: int) -> Any:
        if vtype == _GGUF_TYPE_UINT8:
            return struct.unpack("<B", fp.read(1))[0]
        elif vtype == _GGUF_TYPE_INT8:
            return struct.unpack("<b", fp.read(1))[0]
        elif vtype == _GGUF_TYPE_UINT16:
            return struct.unpack("<H", fp.read(2))[0]
        elif vtype == _GGUF_TYPE_INT16:
            return struct.unpack("<h", fp.read(2))[0]
        elif vtype == _GGUF_TYPE_UINT32:
            return struct.unpack("<I", fp.read(4))[0]
        elif vtype == _GGUF_TYPE_INT32:
            return struct.unpack("<i", fp.read(4))[0]
        elif vtype == _GGUF_TYPE_FLOAT32:
            return struct.unpack("<f", fp.read(4))[0]
        elif vtype == _GGUF_TYPE_BOOL:
            return bool(struct.unpack("<B", fp.read(1))[0])
        elif vtype == _GGUF_TYPE_STRING:
            return self._read_string(fp)
        elif vtype == _GGUF_TYPE_ARRAY:
            arr_type = struct.unpack("<I", fp.read(4))[0]
            arr_len = struct.unpack("<Q", fp.read(8))[0]
            return [self._read_typed_value(fp, arr_type) for _ in range(arr_len)]
        elif vtype == _GGUF_TYPE_UINT64:
            return struct.unpack("<Q", fp.read(8))[0]
        elif vtype == _GGUF_TYPE_INT64:
            return struct.unpack("<q", fp.read(8))[0]
        elif vtype == _GGUF_TYPE_FLOAT64:
            return struct.unpack("<d", fp.read(8))[0]
        else:
            raise ValueError(f"Unknown GGUF value type: {vtype}")

    @property
    def config(self) -> dict[str, Any]:
        """Extract model configuration from metadata."""
        arch = self._metadata.get("general.architecture", "unknown")
        prefix = f"{arch}."
        cfg: dict[str, Any] = {"architecture": arch}

        key_map = {
            "context_length": f"{prefix}context_length",
            "n_layers": f"{prefix}block_count",
            "n_heads": f"{prefix}attention.head_count",
            "n_kv_heads": f"{prefix}attention.head_count_kv",
            "hidden_dim": f"{prefix}embedding_length",
            "ffn_dim": f"{prefix}feed_forward_length",
            "vocab_size": f"{prefix}vocab_size",
            "rope_dim": f"{prefix}rope.dimension_count",
            "rope_freq_base": f"{prefix}rope.freq_base",
            "norm_eps": f"{prefix}attention.layer_norm_rms_epsilon",
        }

        for key, meta_key in key_map.items():
            if meta_key in self._metadata:
                cfg[key] = self._metadata[meta_key]

        # Derive head_dim
        if "hidden_dim" in cfg and "n_heads" in cfg:
            cfg["head_dim"] = cfg["hidden_dim"] // cfg["n_heads"]

        cfg["name"] = self._metadata.get("general.name", "unknown")
        cfg["n_tensors"] = len(self._tensors)

        return cfg

    def tensor_names(self) -> list[str]:
        """List all tensor names."""
        return sorted(self._tensors.keys())

    def tensor_info(self, name: str) -> TensorInfo:
        """Get metadata for a tensor."""
        if name not in self._tensors:
            raise KeyError(f"Tensor not found: {name}. Available: {self.tensor_names()[:10]}...")
        return self._tensors[name]

    def get_tensor_raw(self, name: str) -> bytes:
        """Get raw bytes for a tensor (quantized)."""
        info = self.tensor_info(name)
        size = _TYPE_SIZES.get(info.dtype, 4.0)
        byte_count = int(info.n_elements * size)
        start = self._data_offset + info.offset
        return self._mm[start : start + byte_count]

    def get_tensor(self, name: str) -> NDArray[np.float32]:
        """Load and dequantize a tensor to float32.

        Currently supports F32, F16, and Q8_0. Other quantized types
        return raw bytes as a flat uint8 array (for future dequantization).
        """
        info = self.tensor_info(name)
        start = self._data_offset + info.offset

        if info.dtype == GGML_TYPE_F32:
            byte_count = info.n_elements * 4
            data = np.frombuffer(self._mm[start : start + byte_count], dtype=np.float32)
            return data.reshape(info.shape).copy()

        elif info.dtype == GGML_TYPE_F16:
            byte_count = info.n_elements * 2
            data = np.frombuffer(self._mm[start : start + byte_count], dtype=np.float16)
            return data.reshape(info.shape).astype(np.float32).copy()

        elif info.dtype == GGML_TYPE_Q8_0:
            return self._dequant_q8_0(info)

        elif info.dtype == GGML_TYPE_Q4_K:
            return self._dequant_q4_k(info)

        elif info.dtype in (GGML_TYPE_TQ2, GGML_TYPE_TQ3, GGML_TYPE_TQ4):
            return self._dequant_tq(info)

        else:
            # For other quantized types, return raw — caller can dequantize
            byte_count = int(info.n_elements * _TYPE_SIZES.get(info.dtype, 4.0))
            raw = self._mm[start : start + byte_count]
            return np.frombuffer(raw, dtype=np.uint8).copy()

    def _dequant_q8_0(self, info: TensorInfo) -> NDArray[np.float32]:
        """Dequantize Q8_0: each block = 1 float32 scale + 32 int8 values."""
        start = self._data_offset + info.offset
        block_size = 32
        n_blocks = info.n_elements // block_size
        bytes_per_block = 4 + 32  # float32 scale + 32 int8

        result = np.empty(info.n_elements, dtype=np.float32)
        for i in range(n_blocks):
            offset = start + i * bytes_per_block
            scale = struct.unpack("<f", self._mm[offset : offset + 4])[0]
            quants = np.frombuffer(self._mm[offset + 4 : offset + bytes_per_block], dtype=np.int8)
            result[i * block_size : (i + 1) * block_size] = quants.astype(np.float32) * scale

        return result.reshape(info.shape)

    def _dequant_q4_k(self, info: TensorInfo) -> NDArray[np.float32]:
        """Dequantize Q4_K_M: super-blocks of 256 elements.

        Q4_K layout per super-block (144 bytes for 256 elements):
        - 2 x float16 scales (d, dmin) = 4 bytes
        - 12 bytes of packed 6-bit sub-block scales/mins (k_scales)
        - 128 bytes of packed 4-bit quantized values (qs)
        Total: 144 bytes per 256 elements

        Reference: ggml-common.h block_q4_K definition.
        """
        start = self._data_offset + info.offset
        super_block_size = 256
        n_blocks = info.n_elements // super_block_size
        bytes_per_block = 144  # sizeof(block_q4_K)

        result = np.empty(info.n_elements, dtype=np.float32)

        for i in range(n_blocks):
            block_start = start + i * bytes_per_block

            # Super-block scales: d (float16) and dmin (float16)
            d = np.frombuffer(self._mm[block_start : block_start + 2], dtype=np.float16).astype(np.float32)[0]
            dmin = np.frombuffer(self._mm[block_start + 2 : block_start + 4], dtype=np.float16).astype(np.float32)[0]

            # Sub-block scales and mins packed in 12 bytes (k_scales)
            k_scales_raw = np.frombuffer(self._mm[block_start + 4 : block_start + 16], dtype=np.uint8)

            # Decode 6-bit sub-block scales and mins (8 sub-blocks of 32 elements)
            scales = np.zeros(8, dtype=np.float32)
            mins = np.zeros(8, dtype=np.float32)
            for j in range(8):
                if j < 4:
                    sc = k_scales_raw[j] & 0x3F
                    mn = k_scales_raw[j + 4] & 0x3F
                else:
                    sc = (k_scales_raw[j + 4] & 0x0F) | ((k_scales_raw[j - 4] >> 6) << 4)
                    mn = (k_scales_raw[j + 4] >> 4) | ((k_scales_raw[j] >> 6) << 4)
                scales[j] = d * sc
                mins[j] = dmin * mn

            # Quantized values: 128 bytes of packed 4-bit (256 values)
            qs = np.frombuffer(self._mm[block_start + 16 : block_start + 144], dtype=np.uint8)

            # Unpack: each byte holds two 4-bit values (lo first, hi second)
            out_offset = i * super_block_size
            for j in range(8):
                sub_start = j * 32
                if j < 4:
                    q_bytes = qs[j * 16 : j * 16 + 16]
                else:
                    q_bytes = qs[(j - 4) * 16 + 64 : (j - 4) * 16 + 80]
                lo = (q_bytes & 0x0F).astype(np.float32)
                hi = ((q_bytes >> 4) & 0x0F).astype(np.float32)
                # Sequential layout: first 16 from lo nibbles, next 16 from hi
                vals = np.empty(32, dtype=np.float32)
                vals[:16] = lo
                vals[16:] = hi

                result[out_offset + sub_start : out_offset + sub_start + 32] = vals * scales[j] - mins[j]

        return result.reshape(info.shape)

    def _dequant_tq(self, info: TensorInfo) -> NDArray[np.float32]:
        """Dequantize TurboQuant compressed tensor."""
        from morie.quant import TQBlock, turboquant_mse_decode

        start = self._data_offset + info.offset
        pos = start

        n_elements = struct.unpack("<I", self._mm[pos : pos + 4])[0]
        pos += 4
        n_blocks = struct.unpack("<I", self._mm[pos : pos + 4])[0]
        pos += 4
        bits = struct.unpack("<I", self._mm[pos : pos + 4])[0]
        pos += 4

        block_size = 256
        index_bytes_per_block = (block_size * bits + 7) // 8

        chunks = []
        for _ in range(n_blocks):
            norm = struct.unpack("<f", self._mm[pos : pos + 4])[0]
            pos += 4
            seed = struct.unpack("<H", self._mm[pos : pos + 2])[0]
            pos += 2
            raw_idx = np.frombuffer(self._mm[pos : pos + index_bytes_per_block], dtype=np.uint8).copy()
            pos += index_bytes_per_block

            all_bits = np.unpackbits(raw_idx, bitorder="little")[: block_size * bits]
            indices = np.zeros(block_size, dtype=np.uint8)
            for j in range(block_size):
                val = 0
                for b in range(bits):
                    val |= int(all_bits[j * bits + b]) << b
                indices[j] = val

            block = TQBlock(
                d=block_size,
                bits=bits,
                radius=norm,
                angle_indices=[indices],
                rotation_seed=seed,
            )
            chunks.append(turboquant_mse_decode(block))

        flat = np.concatenate(chunks)[:n_elements]
        return flat.reshape(info.shape).astype(np.float32)

    def close(self) -> None:
        if self._mm:
            try:
                self._mm.close()
            except Exception:
                pass
            self._mm = None
        if self._fp:
            try:
                self._fp.close()
            except Exception:
                pass
            self._fp = None

    def __del__(self):
        self.close()

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"GGUFModel('{self.path.name}', "
            f"arch={cfg.get('architecture')}, "
            f"layers={cfg.get('n_layers')}, "
            f"tensors={cfg.get('n_tensors')})"
        )
