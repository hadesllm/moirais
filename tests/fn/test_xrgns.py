"""Tests for moirais.fn.xrgns -- General Nesting Spatial model"""

import numpy as np
import pytest

from moirais.fn.xrgns import gns_ml


class TestGnsMl:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gns_ml(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = gns_ml(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
