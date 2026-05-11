"""Tests for morie.fn.kvcmp — KV-cache compression."""

import numpy as np
import pytest

from morie.fn.kvcmp import kv_cache_compress


class TestKvCacheCompress:

    def test_returns_result(self):
        rng = np.random.default_rng(42)
        K = rng.standard_normal((2, 4, 16))
        V = rng.standard_normal((2, 4, 16))
        res = kv_cache_compress(K, V, bits=4)
        assert res.name == "kv_cache_compress"

    def test_cosine_high_at_8bit(self):
        rng = np.random.default_rng(0)
        K = rng.standard_normal((1, 2, 32))
        V = rng.standard_normal((1, 2, 32))
        res = kv_cache_compress(K, V, bits=8)
        assert res.value > 0.99

    def test_2d_input(self):
        rng = np.random.default_rng(1)
        K = rng.standard_normal((4, 16))
        V = rng.standard_normal((4, 16))
        res = kv_cache_compress(K, V, bits=3)
        assert res.extra["compression_ratio"] == pytest.approx(32.0 / 3)
