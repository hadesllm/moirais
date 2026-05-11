"""Tests for morie.engine and morie.tokenizer."""

import struct
import numpy as np
import pytest

from morie.engine import (
    MORIEEngine,
    GenerationResult,
    _rmsnorm,
    _softmax,
    _silu,
    _rope,
    backend,
)
from morie.tokenizer import Tokenizer
from morie.gguf_loader import (
    GGML_TYPE_F32,
    GGUF_MAGIC,
    _GGUF_TYPE_ARRAY,
    _GGUF_TYPE_FLOAT32,
    _GGUF_TYPE_INT32,
    _GGUF_TYPE_STRING,
    _GGUF_TYPE_UINT32,
    GGUFModel,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic GGUF files with tokenizer metadata
# ---------------------------------------------------------------------------

def _write_string(buf, s):
    encoded = s.encode("utf-8")
    buf += struct.pack("<Q", len(encoded))
    buf += encoded


def _write_kv(buf, key, vtype, value):
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


def _write_kv_array(buf, key, arr_type, values):
    """Write an array metadata value."""
    _write_string(buf, key)
    buf += struct.pack("<I", _GGUF_TYPE_ARRAY)
    buf += struct.pack("<I", arr_type)
    buf += struct.pack("<Q", len(values))
    for v in values:
        if arr_type == _GGUF_TYPE_STRING:
            _write_string(buf, v)
        elif arr_type == _GGUF_TYPE_FLOAT32:
            buf += struct.pack("<f", v)
        elif arr_type == _GGUF_TYPE_UINT32:
            buf += struct.pack("<I", v)


def _build_gguf_with_tokenizer(tmp_path, vocab=None):
    """Build a minimal GGUF with tokenizer metadata."""
    if vocab is None:
        vocab = ["<unk>", "<s>", "</s>", "▁Hello", "▁world", "▁the", "!"]

    buf = bytearray()
    buf += struct.pack("<I", GGUF_MAGIC)
    buf += struct.pack("<I", 3)

    # We'll add tensors=0, metadata count calculated
    n_tensors = 0
    metadata = []

    # Architecture
    metadata.append(("general.architecture", _GGUF_TYPE_STRING, "llama"))

    # Tokenizer metadata
    # tokens array
    metadata_kv_count = 1 + 4  # arch + 4 tokenizer keys

    buf += struct.pack("<Q", n_tensors)
    buf += struct.pack("<Q", metadata_kv_count)

    # Write architecture
    _write_kv(buf, "general.architecture", _GGUF_TYPE_STRING, "llama")

    # Tokenizer tokens (string array)
    _write_kv_array(buf, "tokenizer.ggml.tokens", _GGUF_TYPE_STRING, vocab)

    # Scores (float32 array)
    scores = [0.0] * len(vocab)
    _write_kv_array(buf, "tokenizer.ggml.scores", _GGUF_TYPE_FLOAT32, scores)

    # BOS and EOS IDs
    _write_kv(buf, "tokenizer.ggml.bos_token_id", _GGUF_TYPE_UINT32, 1)
    _write_kv(buf, "tokenizer.ggml.eos_token_id", _GGUF_TYPE_UINT32, 2)

    # Align
    pos = len(buf)
    pad = (32 - (pos % 32)) % 32
    buf += b"\x00" * pad

    path = tmp_path / "tok_model.gguf"
    path.write_bytes(bytes(buf))
    return path


# ---------------------------------------------------------------------------
# Backend tests
# ---------------------------------------------------------------------------

class TestBackend:
    def test_backend_is_string(self):
        assert backend() in ("mlx", "numpy")


# ---------------------------------------------------------------------------
# Math primitives
# ---------------------------------------------------------------------------

class TestMathPrimitives:
    def test_rmsnorm_shape(self):
        x = np.random.randn(128).astype(np.float32)
        w = np.ones(128, dtype=np.float32)
        result = _rmsnorm(x, w)
        assert result.shape == (128,)

    def test_rmsnorm_normalizes(self):
        x = np.array([3.0, 4.0], dtype=np.float32)
        w = np.ones(2, dtype=np.float32)
        result = _rmsnorm(x, w)
        rms = np.sqrt(np.mean(result ** 2))
        assert abs(rms - 1.0) < 0.01

    def test_softmax_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _softmax(x)
        assert abs(np.sum(result) - 1.0) < 1e-5

    def test_softmax_monotone(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _softmax(x)
        assert result[2] > result[1] > result[0]

    def test_silu_zero(self):
        x = np.array([0.0], dtype=np.float32)
        result = _silu(x)
        assert abs(result[0]) < 1e-5

    def test_silu_positive(self):
        x = np.array([2.0], dtype=np.float32)
        result = _silu(x)
        expected = 2.0 * (1.0 / (1.0 + np.exp(-2.0)))
        assert abs(float(result[0]) - expected) < 1e-5

    def test_rope_preserves_norm(self):
        q = np.random.randn(64).astype(np.float32)
        k = np.random.randn(64).astype(np.float32)
        q_r, k_r = _rope(q, k, position=5, head_dim=64)
        # RoPE is an orthogonal rotation — should preserve norms
        assert abs(np.linalg.norm(q_r) - np.linalg.norm(q)) < 0.01
        assert abs(np.linalg.norm(k_r) - np.linalg.norm(k)) < 0.01

    def test_rope_different_positions(self):
        q = np.ones(64, dtype=np.float32)
        k = np.ones(64, dtype=np.float32)
        q0, _ = _rope(q, k, position=0, head_dim=64)
        q1, _ = _rope(q, k, position=1, head_dim=64)
        # Different positions should give different results
        assert not np.allclose(q0, q1)


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_load_from_gguf(self, tmp_path):
        path = _build_gguf_with_tokenizer(tmp_path)
        model = GGUFModel(path)
        tok = Tokenizer(gguf_model=model)
        assert tok.vocab_size == 7
        assert tok.bos_id == 1
        assert tok.eos_id == 2
        model.close()

    def test_load_from_path(self, tmp_path):
        path = _build_gguf_with_tokenizer(tmp_path)
        tok = Tokenizer(model_path=path)
        assert tok.vocab_size == 7

    def test_decode_basic(self, tmp_path):
        path = _build_gguf_with_tokenizer(tmp_path)
        tok = Tokenizer(model_path=path)
        # Decode known token IDs
        text = tok.decode([3, 4])  # "▁Hello" "▁world" -> "Hello world"
        assert "Hello" in text
        assert "world" in text

    def test_encode_adds_bos(self, tmp_path):
        path = _build_gguf_with_tokenizer(tmp_path)
        tok = Tokenizer(model_path=path)
        ids = tok.encode("Hello", add_bos=True)
        assert ids[0] == tok.bos_id

    def test_encode_no_bos(self, tmp_path):
        path = _build_gguf_with_tokenizer(tmp_path)
        tok = Tokenizer(model_path=path)
        ids_bos = tok.encode("Hello", add_bos=True)
        ids_no_bos = tok.encode("Hello", add_bos=False)
        # With BOS should be longer (or equal if token happens to match BOS)
        assert len(ids_bos) >= len(ids_no_bos)
        # With BOS, first token is BOS
        assert ids_bos[0] == tok.bos_id

    def test_repr(self, tmp_path):
        path = _build_gguf_with_tokenizer(tmp_path)
        tok = Tokenizer(model_path=path)
        r = repr(tok)
        assert "vocab_size=7" in r
        assert "gguf" in r

    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="Provide either"):
            Tokenizer()


# ---------------------------------------------------------------------------
# GenerationResult
# ---------------------------------------------------------------------------

class TestGenerationResult:
    def test_fields(self):
        r = GenerationResult(
            text="hello", tokens=[1, 2], n_tokens=1,
            time_seconds=0.5, tokens_per_second=2.0,
            kv_compression_ratio=4.9, backend="numpy",
        )
        assert r.text == "hello"
        assert r.backend == "numpy"
        assert r.kv_compression_ratio == 4.9


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

class TestSampling:
    def test_greedy(self):
        logits = np.array([0.1, 0.2, 5.0, 0.3])
        token = MORIEEngine._sample(logits, temperature=0.0, top_p=1.0)
        assert token == 2  # argmax

    def test_temperature_affects_distribution(self):
        logits = np.array([1.0, 2.0, 3.0])
        np.random.seed(42)
        # Low temperature -> more deterministic
        counts_low = np.zeros(3)
        for _ in range(100):
            t = MORIEEngine._sample(logits, temperature=0.01, top_p=1.0)
            counts_low[t] += 1
        # Should strongly prefer index 2
        assert counts_low[2] > 90

    def test_top_p_filtering(self):
        logits = np.array([10.0, 0.0, 0.0, 0.0])
        token = MORIEEngine._sample(logits, temperature=0.5, top_p=0.5)
        # With top_p=0.5, only the dominant token should be sampled
        assert token == 0
