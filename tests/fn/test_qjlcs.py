"""Tests for moirais.fn.qjlcs — QJL cosine similarity."""

import numpy as np
import pytest

from moirais.fn.qjlcs import qjl_cosine_sim


class TestQjlCosineSim:

    def test_identical_vectors(self):
        x = np.random.default_rng(42).standard_normal(64)
        res = qjl_cosine_sim(x, x, d_proj=256)
        assert res.value == pytest.approx(1.0, abs=0.05)

    def test_orthogonal_vectors(self):
        x = np.zeros(64)
        x[0] = 1.0
        y = np.zeros(64)
        y[1] = 1.0
        res = qjl_cosine_sim(x, y, d_proj=512)
        assert abs(res.value) < 0.3

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            qjl_cosine_sim(np.ones(10), np.ones(20))
