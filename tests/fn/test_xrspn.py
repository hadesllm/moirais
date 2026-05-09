"""Tests for moirais.fn.xrspn -- Spatial panel fixed effects"""

import numpy as np
import pytest

from moirais.fn.xrspn import spatial_panel_fe


class TestSpatialPanelFe:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_panel_fe(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = spatial_panel_fe(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
