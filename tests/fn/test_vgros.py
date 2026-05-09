"""Tests for moirais.fn.vgros -- Variogram rose diagram"""

import numpy as np
import pytest

from moirais.fn.vgros import vario_rose


class TestVarioRose:
    def test_basic(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        values = rng.standard_normal(20)
        result = vario_rose(coords, values)
        assert result.statistic is not None

    def test_output_type(self):
        result = vario_rose(np.random.default_rng(0).uniform(0,1,(5,2)), np.ones(5))
        assert hasattr(result, "statistic")
