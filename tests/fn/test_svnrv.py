"""Tests for morie.fn.svnrv -- Normal vector to cutting line"""

import numpy as np
import pytest

from morie.fn.svnrv import normal_vector


class TestNormalVector:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normal_vector(data)
        assert result.value is not None

    def test_output_type(self):
        result = normal_vector(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
