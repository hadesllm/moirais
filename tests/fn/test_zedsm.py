"""Tests for morie.fn.zedsm -- Poisson disease mapping"""

import numpy as np
import pytest

from morie.fn.zedsm import disease_map_pois


class TestDiseaseMapPois:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = disease_map_pois(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = disease_map_pois(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
