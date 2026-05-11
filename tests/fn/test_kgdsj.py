"""Tests for morie.fn.kgdsj -- Disjunctive kriging"""

import numpy as np
import pytest

from morie.fn.kgdsj import disjunctive_kriging


class TestDisjunctiveKriging:
    def test_basic(self):
        vals = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = disjunctive_kriging(vals, x)
        assert result.statistic is not None

    def test_output_type(self):
        result = disjunctive_kriging(np.array([1.,2.,3.]), np.array([0.,1.,2.]))
        assert hasattr(result, "statistic")
