"""Tests for morie.fn.ptipo -- Inhomogeneous Poisson process"""

import numpy as np
import pytest

from morie.fn.ptipo import inhom_poisson


class TestInhomPoisson:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = inhom_poisson(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = inhom_poisson(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
