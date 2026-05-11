"""Tests for morie.fn.bb8 -- bootstrap CI alias (BB-8)."""

import numpy as np
import pandas as pd
import pytest
from morie.fn.bb8 import bootstrap_ci as bb8


class TestBB8:
    def test_is_callable(self):
        """bb8 should be a callable (alias for boot.bootstrap_ci)."""
        assert callable(bb8)

    def test_mean_ci(self):
        """BB-8 should compute bootstrap CI for mean."""
        data = pd.DataFrame({"x": np.random.default_rng(42).normal(10.0, 2.0, 100)})
        lo, hi = bb8(lambda df: df["x"].mean(), data, n_iterations=200)
        assert lo < 10.0 < hi

    def test_same_as_boot(self):
        """BB-8 alias should give identical results to boot.bootstrap_ci."""
        from morie.fn.boot import bootstrap_ci
        data = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        r1 = bb8(lambda df: df["x"].mean(), data, seed=99)
        r2 = bootstrap_ci(lambda df: df["x"].mean(), data, seed=99)
        assert r1[0] == pytest.approx(r2[0])
        assert r1[1] == pytest.approx(r2[1])
