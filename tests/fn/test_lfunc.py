"""Tests for moirais.fn.lfunc — Ripley's L."""
import numpy as np
from moirais.fn.lfunc import ripley_l


class TestRipleyL:
    def test_basic(self):
        pts = np.random.default_rng(42).uniform(0, 10, (30, 2))
        res = ripley_l(pts, n_distances=10)
        assert len(res.extra["L"]) == 10

    def test_has_k(self):
        pts = np.random.default_rng(42).uniform(0, 10, (20, 2))
        res = ripley_l(pts)
        assert "K" in res.extra
