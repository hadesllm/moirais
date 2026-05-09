"""Tests for moirais.fn.ze3sf -- Three-step FCA"""

import numpy as np
import pytest

from moirais.fn.ze3sf import three_step_fca


class TestThreeStepFca:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = three_step_fca(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = three_step_fca(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
