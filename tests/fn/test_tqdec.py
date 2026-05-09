"""Tests for moirais.fn.tqdec — TurboQuant dequantization."""

import numpy as np
import pytest

from moirais.fn.tqmse import turboquant_mse
from moirais.fn.tqdec import turboquant_decode


class TestTurboquantDecode:

    def test_roundtrip(self):
        x = np.random.default_rng(42).standard_normal(16)
        enc = turboquant_mse(x, bits=4)
        dec = turboquant_decode(
            enc.extra["codes"],
            enc.extra["centroids"],
            n_original=enc.extra["n_original"],
            bits=4,
        )
        assert dec.name == "turboquant_decode"
        assert len(dec.extra["reconstructed"]) == 16

    def test_returns_array(self):
        codes = np.array([0, 1, 2, 3], dtype=np.int32)
        centroids = np.array([-1.0, -0.3, 0.3, 1.0])
        res = turboquant_decode(codes, centroids, n_original=4)
        assert len(res.extra["reconstructed"]) == 4
