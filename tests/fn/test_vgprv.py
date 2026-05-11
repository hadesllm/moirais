"""Tests for morie.fn.vgprv -- Pairwise relative variogram"""

import numpy as np
import pytest

from morie.fn.vgprv import pairwise_rel_vario


class TestPairwiseRelVario:
    def test_basic(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        values = rng.standard_normal(20)
        result = pairwise_rel_vario(coords, values)
        assert result.statistic is not None

    def test_output_type(self):
        result = pairwise_rel_vario(np.random.default_rng(0).uniform(0,1,(5,2)), np.ones(5))
        assert hasattr(result, "statistic")
