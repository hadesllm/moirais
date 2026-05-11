"""Tests for morie.fn.mssm2 -- SMACOF 2D MDS"""

import numpy as np
import pytest

from morie.fn.mssm2 import smacof_2d


class TestSmacof2d:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = smacof_2d(data)
        assert result.value is not None

    def test_output_type(self):
        result = smacof_2d(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
