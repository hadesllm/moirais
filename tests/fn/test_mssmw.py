"""Tests for morie.fn.mssmw -- Weighted SMACOF MDS"""

import numpy as np
import pytest

from morie.fn.mssmw import smacof_weight


class TestSmacofWeight:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = smacof_weight(data)
        assert result.value is not None

    def test_output_type(self):
        result = smacof_weight(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
