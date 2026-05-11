"""Tests for morie.fn.kgdsh -- Disjunctive kriging Hermite polynomials"""

import numpy as np
import pytest

from morie.fn.kgdsh import dk_hermite


class TestDkHermite:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dk_hermite(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = dk_hermite(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
