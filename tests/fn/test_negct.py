"""Tests for moirais.fn.negct — Negative control test."""

import pytest

from moirais.fn.negct import negative_control


class TestNegativeControl:
    def test_no_confounding(self):
        res = negative_control(0.5, 0.01, se_estimate=0.1, se_negative=0.1)
        assert res.p_value > 0.05

    def test_confounding_suspected(self):
        res = negative_control(0.5, 0.8, se_estimate=0.1, se_negative=0.1)
        assert res.extra["confounding_suspected"] is True

    def test_returns_test_result(self):
        res = negative_control(1.0, 0.5)
        assert res.test_name == "negative_control"
