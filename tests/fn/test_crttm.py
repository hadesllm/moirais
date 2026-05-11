"""Tests for morie.fn.crttm — time to trial."""

import pytest
import numpy as np
from morie.fn.crttm import court_time_to_trial
from morie.fn._containers import DescriptiveResult


class TestTimeToTrial:
    def test_basic(self):
        rng = np.random.default_rng(42)
        r = court_time_to_trial(rng.exponential(180, 500))
        assert isinstance(r, DescriptiveResult)
        assert r.extra["n"] == 500

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            court_time_to_trial([])
