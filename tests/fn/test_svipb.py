"""Tests for morie.fn.svipb -- Bayesian ideal point posterior"""

import numpy as np
import pytest

from morie.fn.svipb import ideal_point_bayes


class TestIdealPointBayes:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ideal_point_bayes(data)
        assert result.value is not None

    def test_output_type(self):
        result = ideal_point_bayes(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
