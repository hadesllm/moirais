"""Tests for moirais.fn.sveln -- Elbow method for dimensions"""

import numpy as np
import pytest

from moirais.fn.sveln import elbow_spatial


class TestElbowSpatial:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = elbow_spatial(data)
        assert result.value is not None

    def test_output_type(self):
        result = elbow_spatial(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
