"""Tests for moirais.fn.ptkdb -- KDE bandwidth selection (spatial)"""

import numpy as np
import pytest

from moirais.fn.ptkdb import kde_bandwidth


class TestKdeBandwidth:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = kde_bandwidth(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = kde_bandwidth(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
