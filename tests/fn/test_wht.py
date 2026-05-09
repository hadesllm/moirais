"""Tests for moirais.fn.wht — Walsh-Hadamard Transform."""

import numpy as np
import pytest

from moirais.fn.wht import walsh_hadamard


class TestWalshHadamard:

    def test_returns_result(self):
        x = np.random.default_rng(42).standard_normal(16)
        res = walsh_hadamard(x)
        assert res.name == "walsh_hadamard"
        assert res.extra["normalization"] == "1/sqrt(d)"

    def test_involution(self):
        """WHT with 1/sqrt(d) is its own inverse."""
        x = np.random.default_rng(0).standard_normal(16)
        h = walsh_hadamard(x).extra["full"]
        x_back = walsh_hadamard(h).extra["full"]
        assert np.allclose(x_back[:16], x, atol=1e-10)

    def test_norm_preservation(self):
        """Orthonormal WHT preserves L2 norm (Parseval)."""
        x = np.random.default_rng(1).standard_normal(32)
        res = walsh_hadamard(x)
        h = res.extra["full"]
        assert np.linalg.norm(h) == pytest.approx(np.linalg.norm(
            np.concatenate([x, np.zeros(32 - len(x))]) if len(x) < 32 else x
        ), abs=1e-10)

    def test_padding(self):
        x = np.random.default_rng(2).standard_normal(10)
        res = walsh_hadamard(x)
        assert res.extra["d_padded"] == 16
        assert res.extra["d_original"] == 10
