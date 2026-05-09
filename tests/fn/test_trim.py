"""Tests for moirais.fn.trim -- Propensity score trimming."""

import numpy as np
import pytest
from moirais.fn.trim import ps_trim


class TestPSTrim:
    def test_keeps_middle_scores(self):
        ps = np.array([0.05, 0.15, 0.5, 0.85, 0.95])
        mask = ps_trim(ps, threshold=0.1)
        assert mask.tolist() == [False, True, True, True, False]

    def test_all_kept_when_in_range(self):
        ps = np.array([0.2, 0.3, 0.5, 0.7, 0.8])
        mask = ps_trim(ps, threshold=0.1)
        assert all(mask)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            ps_trim([0.5], threshold=0.6)

    def test_returns_boolean_array(self):
        ps = np.linspace(0, 1, 20)
        mask = ps_trim(ps, threshold=0.1)
        assert mask.dtype == bool
