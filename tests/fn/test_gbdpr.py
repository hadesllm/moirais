"""Tests for morie.fn.gbdpr -- GBD projection."""

import pytest
from morie.fn.gbdpr import gbd_projection


class TestGBDProjection:
    def test_basic(self):
        res = gbd_projection(current_burden=1000, growth_rate=0.05, years=10)
        assert res.value == pytest.approx(1000 * 1.05 ** 10)

    def test_zero_years(self):
        res = gbd_projection(1000, 0.05, 0)
        assert res.value == pytest.approx(1000.0)
