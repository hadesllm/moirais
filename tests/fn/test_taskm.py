"""Tests for morie.fn.taskm -- dynamic time warping."""

import numpy as np
from morie.fn.taskm import dtw_match, taskm
from morie.fn._containers import DescriptiveResult


class TestTaskm:
    def test_alias(self):
        assert taskm is dtw_match

    def test_identical(self):
        x = np.sin(np.linspace(0, 2 * np.pi, 50))
        r = dtw_match(x, x)
        assert isinstance(r, DescriptiveResult)
        assert r.value["distance"] < 1e-10

    def test_shifted(self):
        t = np.linspace(0, 2 * np.pi, 50)
        q = np.sin(t)
        template = np.sin(t + 0.5)
        r = dtw_match(q, template)
        assert r.value["distance"] > 0
        assert r.value["path_length"] >= 50
