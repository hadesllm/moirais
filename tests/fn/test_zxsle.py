"""Tests for moirais.fn.zxsle -- Spatial elastic net"""

import numpy as np
import pytest

from moirais.fn.zxsle import spatial_elastic


class TestSpatialElastic:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_elastic(data)
        assert result.value is not None

    def test_output_type(self):
        result = spatial_elastic(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
