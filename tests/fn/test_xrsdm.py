"""Tests for moirais.fn.xrsdm -- Spatial Durbin model ML"""

import numpy as np
import pytest

from moirais.fn.xrsdm import sdm_ml


class TestSdmMl:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sdm_ml(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = sdm_ml(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
