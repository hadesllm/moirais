"""Tests for morie.fn.xrspr -- Spatial panel random effects"""

import numpy as np
import pytest

from morie.fn.xrspr import spatial_panel_re


class TestSpatialPanelRe:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_panel_re(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = spatial_panel_re(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
