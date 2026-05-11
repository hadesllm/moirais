"""Tests for morie.fn.msref -- Reflection of configuration"""

import numpy as np
import pytest

from morie.fn.msref import reflect_config


class TestReflectConfig:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = reflect_config(data)
        assert result.value is not None

    def test_output_type(self):
        result = reflect_config(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
