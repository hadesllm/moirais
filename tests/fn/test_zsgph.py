"""Tests for moirais.fn.zsgph -- GP hyperparameter optimization"""

import numpy as np
import pytest

from moirais.fn.zsgph import gp_hyperparams


class TestGpHyperparams:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gp_hyperparams(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = gp_hyperparams(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
