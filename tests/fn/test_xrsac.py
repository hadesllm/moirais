"""Tests for moirais.fn.xrsac -- SAC/SARAR model ML"""

import numpy as np
import pytest

from moirais.fn.xrsac import sac_ml


class TestSacMl:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sac_ml(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = sac_ml(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
