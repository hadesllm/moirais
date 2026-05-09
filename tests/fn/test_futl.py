"""Tests for moirais.fn.futl — Futility boundary."""

import pytest

from moirais.fn.futl import futility_boundary


class TestFutilityBoundary:
    def test_basic(self):
        res = futility_boundary(4)
        assert len(res.extra["boundaries"]) == 4

    def test_final_boundary(self):
        res = futility_boundary(3, power=0.90)
        assert res.extra["boundaries"][-1] > res.extra["boundaries"][0]

    def test_too_few(self):
        with pytest.raises(ValueError):
            futility_boundary(1)
