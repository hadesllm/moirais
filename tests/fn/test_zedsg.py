"""Tests for moirais.fn.zedsg -- Poisson-Gamma disease mapping"""

import numpy as np
import pytest

from moirais.fn.zedsg import disease_map_gamma


class TestDiseaseMapGamma:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = disease_map_gamma(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = disease_map_gamma(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
