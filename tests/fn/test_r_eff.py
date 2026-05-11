"""Tests for morie.fn.r_eff -- effective reproduction number."""

import numpy as np
import pytest
from morie.fn.r_eff import effective_rt


class TestReff:
    def test_growing_epidemic(self):
        """Exponentially growing incidence should give Rt > 1."""
        inc = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], dtype=float)
        res = effective_rt(inc, serial_interval=1.0, window=1)
        rt = res.value
        assert np.nanmean(rt) > 1.0

    def test_declining_epidemic(self):
        """Declining incidence should give Rt < 1."""
        inc = np.array([1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1], dtype=float)
        res = effective_rt(inc, serial_interval=1.0, window=1)
        rt = res.value
        assert np.nanmean(rt) < 1.0

    def test_short_series_raises(self):
        """Series shorter than window should raise."""
        with pytest.raises(ValueError):
            effective_rt(np.array([1, 2, 3]), window=5)
