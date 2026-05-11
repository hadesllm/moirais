"""Tests for morie.fn.mssmi -- Individual differences SMACOF"""

import numpy as np
import pytest

from morie.fn.mssmi import smacof_indiv


class TestSmacofIndiv:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = smacof_indiv(data)
        assert result.value is not None

    def test_output_type(self):
        result = smacof_indiv(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
