"""Tests for moirais.fn.ptdlr -- Delaunay residuals"""

import numpy as np
import pytest

from moirais.fn.ptdlr import pp_delaunay_resid


class TestPpDelaunayResid:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pp_delaunay_resid(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = pp_delaunay_resid(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
