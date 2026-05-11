"""Tests for morie.fn.mssmm -- SMACOF with missing data"""

import numpy as np
import pytest

from morie.fn.mssmm import smacof_missing


class TestSmacofMissing:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = smacof_missing(data)
        assert result.value is not None

    def test_output_type(self):
        result = smacof_missing(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
