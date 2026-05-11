"""Tests for morie.fn.svsl2 -- Two-issue salience model"""

import numpy as np
import pytest

from morie.fn.svsl2 import salience_2issue


class TestSalience2issue:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = salience_2issue(data)
        assert result.value is not None

    def test_output_type(self):
        result = salience_2issue(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
