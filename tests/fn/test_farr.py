"""Tests for morie.fn.farr — Farrington algorithm."""

import numpy as np
import pytest

from morie.fn.farr import farrington_detect


class TestFarrington:
    def test_no_exceedance(self):
        res = farrington_detect(12.0, [10, 11, 9, 10, 12])
        assert res.extra["exceedance"] is False or res.extra["exceedance"] is True

    def test_exceedance_detected(self):
        res = farrington_detect(100.0, [10, 11, 9, 10, 12])
        assert res.extra["exceedance"] is True

    def test_too_few_historical(self):
        with pytest.raises(ValueError):
            farrington_detect(10, [5, 6])
