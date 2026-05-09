"""Tests for moirais.gguf_loader — GGUF model file parser."""

import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from moirais.gguf_loader import (
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_Q8_0,
    GGUF_MAGIC,
    GGUFModel,
    TensorInfo,
    _GGUF_TYPE_FLOAT32,
    _GGUF_TYPE_INT32,
    _GGUF_TYPE_STRING,
    _GGUF_TYPE_UINT32,
)


def _write_string(buf: bytearray, s: str) -> None:
    """Write a GGUF string (uint64 length + utf-8 bytes)."""
    encoded = s.encode("utf-8")
    buf += struct.pack("<Q", len(encoded))
    buf += encoded


def _write_kv(buf: bytearray, key: str, vtype: int, value) -> None:
    """Write a GGUF metadata key-value pair."""
    _write_string(buf, key)
    buf += struct.pack("<I", vtype)
    if vtype == _GGUF_TYPE_UINT32:
        buf += struct.pack("<I", value)
    elif vtype == _GGUF_TYPE_INT32:
        buf += struct.pack("<i", value)
    elif vtype == _GGUF_TYPE_FLOAT32:
        buf += struct.pack("<f", value)
    elif vtype == _GGUF_TYPE_STRING:
        _write_string(buf, value)


def _build_minimal_gguf(
    tensors: dict[str, tuple[tuple[int, ...], int, np.ndarray]],
    metadata: dict[str, tuple[int, object]] | None = None,
) -> bytes:
    """Build a minimal valid GGUF v3 file in memory.

    Parameters
    ----------
    tensors : dict mapping name -> (shape, ggml_type, data_bytes_as_ndarray)
    metadata : dict mapping key -> (gguf_type, value)
    """
    if metadata is None:
        metadata = {}

    buf = bytearray()
    # Header: magic, version, n_tensors, n_kv
    buf += struct.pack("<I", GGUF_MAGIC)
    buf += struct.pack("<I", 3)  # version 3
    buf += struct.pack("<Q", len(tensors))
    buf += struct.pack("<Q", len(metadata))

    # Metadata
    for key, (vtype, value) in metadata.items():
        _write_kv(buf, key, vtype, value)

    # Tensor info
    tensor_data_parts = []
    offset = 0
    for name, (shape, dtype, data) in tensors.items():
        _write_string(buf, name)
        buf += struct.pack("<I", len(shape))  # n_dims
        for dim in shape:
            buf += struct.pack("<Q", dim)
        buf += struct.pack("<I", dtype)
        buf += struct.pack("<Q", offset)
        tensor_data_parts.append(data.tobytes())
        offset += len(data.tobytes())

    # Align to 32 bytes
    pos = len(buf)
    alignment = 32
    pad = (alignment - (pos % alignment)) % alignment
    buf += b"\x00" * pad

    # Tensor data
    for part in tensor_data_parts:
        buf += part

    return bytes(buf)


# ──────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────


class TestTensorInfo:
    def test_dtype_name_known(self):
        ti = TensorInfo(name="w", shape=(4,), dtype=GGML_TYPE_F32, offset=0, n_elements=4)
        assert ti.dtype_name == "F32"

    def test_dtype_name_unknown(self):
        ti = TensorInfo(name="w", shape=(4,), dtype=99, offset=0, n_elements=4)
        assert "type_99" in ti.dtype_name


class TestGGUFModelF32:
    """Test GGUFModel with a synthetic GGUF file containing F32 tensors."""

    @pytest.fixture
    def model_path(self, tmp_path):
        rng = np.random.default_rng(42)
        w1 = rng.standard_normal(16).astype(np.float32)
        w2 = rng.standard_normal((4, 8)).astype(np.float32)

        data = _build_minimal_gguf(
            tensors={
                "weight_a": ((16,), GGML_TYPE_F32, w1),
                "weight_b": ((4, 8), GGML_TYPE_F32, w2),
            },
            metadata={
                "general.architecture": (_GGUF_TYPE_STRING, "llama"),
                "general.name": (_GGUF_TYPE_STRING, "test-model"),
            },
        )
        path = tmp_path / "test.gguf"
        path.write_bytes(data)
        return path, w1, w2

    def test_loads_and_parses(self, model_path):
        path, _, _ = model_path
        model = GGUFModel(path)
        assert model.config["architecture"] == "llama"
        assert model.config["name"] == "test-model"
        assert model.config["n_tensors"] == 2
        model.close()

    def test_tensor_names(self, model_path):
        path, _, _ = model_path
        model = GGUFModel(path)
        names = model.tensor_names()
        assert "weight_a" in names
        assert "weight_b" in names
        model.close()

    def test_get_tensor_f32_1d(self, model_path):
        path, w1, _ = model_path
        model = GGUFModel(path)
        loaded = model.get_tensor("weight_a")
        np.testing.assert_allclose(loaded, w1, atol=1e-6)
        model.close()

    def test_get_tensor_f32_2d(self, model_path):
        path, _, w2 = model_path
        model = GGUFModel(path)
        loaded = model.get_tensor("weight_b")
        np.testing.assert_allclose(loaded, w2, atol=1e-6)
        model.close()

    def test_tensor_info_shape(self, model_path):
        path, _, _ = model_path
        model = GGUFModel(path)
        info = model.tensor_info("weight_b")
        assert info.shape == (4, 8)
        assert info.dtype == GGML_TYPE_F32
        assert info.n_elements == 32
        model.close()

    def test_missing_tensor_raises(self, model_path):
        path, _, _ = model_path
        model = GGUFModel(path)
        with pytest.raises(KeyError, match="not_here"):
            model.tensor_info("not_here")
        model.close()


class TestGGUFModelF16:
    """Test F16 dequantization."""

    @pytest.fixture
    def model_path(self, tmp_path):
        rng = np.random.default_rng(123)
        w = rng.standard_normal(32).astype(np.float16)

        data = _build_minimal_gguf(
            tensors={"embed": ((32,), GGML_TYPE_F16, w)},
            metadata={"general.architecture": (_GGUF_TYPE_STRING, "llama")},
        )
        path = tmp_path / "f16.gguf"
        path.write_bytes(data)
        return path, w

    def test_f16_roundtrip(self, model_path):
        path, w = model_path
        model = GGUFModel(path)
        loaded = model.get_tensor("embed")
        assert loaded.dtype == np.float32
        np.testing.assert_allclose(loaded, w.astype(np.float32), atol=1e-3)
        model.close()


class TestGGUFModelQ8_0:
    """Test Q8_0 dequantization."""

    @pytest.fixture
    def model_path(self, tmp_path):
        rng = np.random.default_rng(99)
        # Q8_0: blocks of 32 values, each block = float32 scale + 32 int8
        n_blocks = 4
        block_size = 32
        raw_values = rng.standard_normal(n_blocks * block_size).astype(np.float32)

        # Quantize: scale = max(abs(block)) / 127, quants = round(block / scale)
        q8_data = bytearray()
        for i in range(n_blocks):
            block = raw_values[i * block_size : (i + 1) * block_size]
            scale = np.max(np.abs(block)) / 127.0
            quants = np.round(block / scale).clip(-128, 127).astype(np.int8)
            q8_data += struct.pack("<f", scale)
            q8_data += quants.tobytes()

        q8_array = np.frombuffer(bytes(q8_data), dtype=np.uint8)

        data = _build_minimal_gguf(
            tensors={"q8_tensor": ((n_blocks * block_size,), GGML_TYPE_Q8_0, q8_array)},
            metadata={"general.architecture": (_GGUF_TYPE_STRING, "llama")},
        )
        path = tmp_path / "q8.gguf"
        path.write_bytes(data)
        return path, raw_values

    def test_q8_0_dequant_close(self, model_path):
        path, raw_values = model_path
        model = GGUFModel(path)
        loaded = model.get_tensor("q8_tensor")
        assert loaded.shape == raw_values.shape
        # Q8_0 dequant should be close but not exact (quantization error)
        np.testing.assert_allclose(loaded, raw_values, atol=0.05)
        model.close()


class TestGGUFModelErrors:
    def test_bad_magic_raises(self, tmp_path):
        path = tmp_path / "bad.gguf"
        path.write_bytes(b"\x00\x00\x00\x00" + b"\x00" * 100)
        with pytest.raises(ValueError, match="Not a GGUF file"):
            GGUFModel(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            GGUFModel(tmp_path / "nonexistent.gguf")

    def test_unsupported_version_raises(self, tmp_path):
        buf = bytearray()
        buf += struct.pack("<I", GGUF_MAGIC)
        buf += struct.pack("<I", 99)  # bad version
        buf += struct.pack("<Q", 0)
        buf += struct.pack("<Q", 0)
        path = tmp_path / "badver.gguf"
        path.write_bytes(bytes(buf))
        with pytest.raises(ValueError, match="Unsupported GGUF version"):
            GGUFModel(path)


class TestGGUFModelRepr:
    def test_repr(self, tmp_path):
        data = _build_minimal_gguf(
            tensors={},
            metadata={
                "general.architecture": (_GGUF_TYPE_STRING, "test"),
                "general.name": (_GGUF_TYPE_STRING, "mymodel"),
            },
        )
        path = tmp_path / "repr.gguf"
        path.write_bytes(data)
        model = GGUFModel(path)
        r = repr(model)
        assert "test" in r
        assert "tensors=0" in r
        model.close()
