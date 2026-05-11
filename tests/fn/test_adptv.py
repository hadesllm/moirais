"""Tests for morie.fn.adptv — Adaptive design."""

import pytest

from morie.fn.adptv import adaptive_design


class TestAdaptiveDesign:
    def test_small_effect_increases_n(self):
        res = adaptive_design(0.2, 100, sd=1.0)
        assert res.estimate >= 100

    def test_large_effect_keeps_n(self):
        res = adaptive_design(2.0, 100, sd=1.0)
        assert res.estimate <= 200

    def test_zero_effect(self):
        res = adaptive_design(0.0, 100)
        assert res.estimate == 200
