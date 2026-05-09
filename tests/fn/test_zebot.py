"""Tests for moirais.fn.zebot -- Bayesian outbreak detection"""

import numpy as np
import pytest

from moirais.fn.zebot import bayes_outbreak


class TestBayesOutbreak:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bayes_outbreak(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = bayes_outbreak(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
