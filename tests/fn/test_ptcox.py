"""Tests for moirais.fn.ptcox -- Cox (doubly stochastic) process"""

import numpy as np
import pytest

from moirais.fn.ptcox import cox_process


class TestCoxProcess:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = cox_process(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = cox_process(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
