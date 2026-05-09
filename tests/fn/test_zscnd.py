"""Tests for moirais.fn.zscnd -- Conditional simulation"""

import numpy as np
import pytest

from moirais.fn.zscnd import conditional_sim


class TestConditionalSim:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = conditional_sim(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = conditional_sim(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
