"""Tests for morie.fn.ewma — EWMA control chart."""

import numpy as np
import pytest

from morie.fn.ewma import ewma_detect


class TestEWMA:
    def test_no_alarm_baseline(self):
        rng = np.random.default_rng(42)
        counts = rng.poisson(10, 60)
        res = ewma_detect(counts)
        assert isinstance(res.extra["alarm_indices"], list)

    def test_alarm_on_outbreak(self):
        baseline = np.full(30, 10.0)
        outbreak = np.full(10, 50.0)
        counts = np.concatenate([baseline, outbreak])
        res = ewma_detect(counts)
        assert len(res.extra["alarm_indices"]) > 0

    def test_too_short(self):
        with pytest.raises(ValueError):
            ewma_detect([1, 2])
