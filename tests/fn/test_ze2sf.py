"""Tests for morie.fn.ze2sf -- Two-step floating catchment area"""

import numpy as np
import pytest

from morie.fn.ze2sf import two_step_fca


class TestTwoStepFca:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = two_step_fca(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = two_step_fca(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
