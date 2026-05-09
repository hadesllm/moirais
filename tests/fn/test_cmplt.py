"""Tests for moirais.fn.cmplt -- Capture-recapture."""

import pytest
from moirais.fn.cmplt import capture_recapture


class TestCaptureRecapture:
    def test_known(self):
        res = capture_recapture(n1=100, n2=100, m=20)
        assert res.measure == "capture_recapture"
        assert res.estimate > 100

    def test_chapman(self):
        res = capture_recapture(n1=100, n2=100, m=50)
        expected = ((101 * 101) / 51) - 1
        assert res.estimate == pytest.approx(expected, rel=0.01)

    def test_zero_overlap(self):
        with pytest.raises(ValueError):
            capture_recapture(n1=100, n2=100, m=0)
