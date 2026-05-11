"""Tests for morie.fn.vgsph -- Spherical variogram model"""

import numpy as np
import pytest

from morie.fn.vgsph import vario_spherical


class TestVarioSpherical:
    def test_basic(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        values = rng.standard_normal(20)
        result = vario_spherical(coords, values)
        assert result.statistic is not None

    def test_output_type(self):
        result = vario_spherical(np.random.default_rng(0).uniform(0,1,(5,2)), np.ones(5))
        assert hasattr(result, "statistic")
