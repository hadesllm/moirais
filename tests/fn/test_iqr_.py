"""Tests for iqr_statistic."""
import numpy as np, pytest
from moirais.fn.iqr_ import iqr_statistic

class TestIQR:
    def test_known(self):
        x = np.arange(1, 101, dtype=float)
        r = iqr_statistic(x)
        assert r.estimate == pytest.approx(50.0, abs=1.0)

    def test_small(self):
        with pytest.raises(ValueError):
            iqr_statistic(np.array([1.0, 2.0]))
