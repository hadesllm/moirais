"""Tests for morie.fn.tpsrs — response time."""

import pytest
import numpy as np
from morie.fn.tpsrs import tps_response_time
from morie.fn._containers import DescriptiveResult


class TestResponseTime:
    def test_basic(self):
        r = tps_response_time([5, 10, 15, 20, 25])
        assert isinstance(r, DescriptiveResult)
        assert r.value == pytest.approx(15.0)

    def test_percentiles(self):
        rng = np.random.default_rng(42)
        r = tps_response_time(rng.exponential(10, 1000))
        assert r.extra["p90"] > r.extra["median"]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            tps_response_time([])
