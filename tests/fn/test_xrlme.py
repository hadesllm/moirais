"""Tests for moirais.fn.xrlme -- LM test for spatial error"""

import numpy as np
import pytest

from moirais.fn.xrlme import lm_error


class TestLmError:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = lm_error(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = lm_error(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
