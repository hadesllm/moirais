"""Tests for moirais.fn.ptthm -- Thomas cluster process"""

import numpy as np
import pytest

from moirais.fn.ptthm import thomas_process


class TestThomasProcess:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = thomas_process(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = thomas_process(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
