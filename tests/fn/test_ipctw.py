"""Tests for moirais.fn.ipctw — IPCW."""

import numpy as np
import pytest

from moirais.fn.ipctw import ipcw


class TestIPCW:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        c = (rng.uniform(size=n) > 0.3).astype(int)
        res = ipcw(c, x)
        assert res.extra["mean_weight"] > 0

    def test_uncensored_have_weights(self):
        rng = np.random.default_rng(42)
        n = 100
        c = np.ones(n, dtype=int)
        c[:30] = 0
        x = rng.normal(0, 1, n)
        res = ipcw(c, x)
        assert res.extra["n_uncensored"] == 70

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            ipcw([1, 0, 1], [[1], [2]])
