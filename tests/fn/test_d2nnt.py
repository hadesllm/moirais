"""Tests for morie.fn.d2nnt -- Convert Cohen's d to NNT."""

import numpy as np
import pytest
from morie.fn.d2nnt import d_to_nnt


class TestDToNNT:
    def test_moderate_effect(self):
        """d=0.8 with base_rate=0.5 gives NNT around 3-5."""
        nnt = d_to_nnt(0.8)
        assert isinstance(nnt, float)
        assert 2 < nnt < 6

    def test_zero_d_gives_inf(self):
        """d=0 gives NNT = inf (no treatment effect)."""
        nnt = d_to_nnt(0.0)
        assert nnt == np.inf or nnt > 1e10

    def test_large_d_small_nnt(self):
        """Larger d should give smaller NNT."""
        nnt_small = d_to_nnt(0.2)
        nnt_large = d_to_nnt(1.5)
        assert nnt_large < nnt_small
