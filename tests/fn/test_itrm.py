"""Tests for morie.fn.itrm — Interim analysis boundaries."""

import pytest

from morie.fn.itrm import interim_analysis


class TestInterimAnalysis:
    def test_obrien_fleming(self):
        res = interim_analysis(4, method="obrien_fleming")
        assert res.extra["boundaries"][0] > res.extra["boundaries"][-1]

    def test_pocock(self):
        res = interim_analysis(3, method="pocock")
        b = res.extra["boundaries"]
        assert b[0] == pytest.approx(b[1])

    def test_too_few_looks(self):
        with pytest.raises(ValueError):
            interim_analysis(1)
