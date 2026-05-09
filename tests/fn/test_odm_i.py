"""Tests for moirais.fn.odm_i — OTIS demo diversity index."""

import pytest
import numpy as np
from moirais.fn.odm_i import otis_demo_index
from moirais.fn._containers import ESRes


class TestOtisDemoIndex:

    def test_returns_esres(self):
        counts = np.array([30, 40, 30])
        result = otis_demo_index(counts)
        assert isinstance(result, ESRes)
        assert result.estimate > 0

    def test_single_group(self):
        result = otis_demo_index(np.array([100]))
        assert result.estimate == pytest.approx(0.0)

    def test_shannon(self):
        counts = np.array([25, 25, 25, 25])
        result = otis_demo_index(counts, method="shannon")
        assert result.estimate == pytest.approx(np.log(4), rel=0.01)
