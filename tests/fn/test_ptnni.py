"""Tests for moirais.fn.ptnni -- Nearest neighbor index (Clark-Evans)"""

import numpy as np
import pytest

from moirais.fn.ptnni import nn_index


class TestNnIndex:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = nn_index(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = nn_index(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
