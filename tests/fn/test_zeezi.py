"""Tests for moirais.fn.zeezi -- Ecological zero-inflated"""

import numpy as np
import pytest

from moirais.fn.zeezi import ecological_zip


class TestEcologicalZip:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ecological_zip(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = ecological_zip(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
