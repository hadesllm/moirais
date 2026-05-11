"""Tests for morie.fn.abrtm -- Farrington aberration detection."""

import pytest
from morie.fn.abrtm import farrington_aberration


class TestFarrington:
    def test_no_aberration(self):
        counts = [10] * 20
        res = farrington_aberration(counts)
        assert res.estimate == 0

    def test_aberration_detected(self):
        counts = [10] * 15 + [100, 120, 90, 80, 50]
        res = farrington_aberration(counts)
        assert res.estimate > 0

    def test_too_short(self):
        with pytest.raises(ValueError):
            farrington_aberration([1, 2, 3])
