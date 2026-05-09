"""Tests for moirais.fn.msrot -- Rotation of configuration"""

import numpy as np
import pytest

from moirais.fn.msrot import rotate_config


class TestRotateConfig:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rotate_config(data)
        assert result.value is not None

    def test_output_type(self):
        result = rotate_config(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
