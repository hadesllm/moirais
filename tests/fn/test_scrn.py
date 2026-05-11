"""Tests for morie.fn.scrn — Screening test metrics."""

import pytest

from morie.fn.scrn import screening_metrics


class TestScreeningMetrics:
    def test_perfect_test(self):
        res = screening_metrics(50, 0, 0, 50)
        assert res.extra["sensitivity"] == pytest.approx(1.0)
        assert res.extra["specificity"] == pytest.approx(1.0)

    def test_known_values(self):
        res = screening_metrics(80, 20, 10, 90)
        assert res.extra["sensitivity"] == pytest.approx(80 / 90)
        assert res.extra["ppv"] == pytest.approx(80 / 100)

    def test_negative_input(self):
        with pytest.raises(ValueError):
            screening_metrics(-1, 0, 0, 10)
