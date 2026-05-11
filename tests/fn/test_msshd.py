"""Tests for morie.fn.msshd -- Shepard disparities"""

import numpy as np
import pytest

from morie.fn.msshd import shepard_dist


class TestShepardDist:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = shepard_dist(data)
        assert result.value is not None

    def test_output_type(self):
        result = shepard_dist(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
