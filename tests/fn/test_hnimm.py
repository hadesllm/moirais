"""Tests for morie.fn.hnimm -- herd immunity threshold."""

import pytest
from morie.fn.hnimm import herd_immunity_threshold


class TestHerdImmunity:
    def test_measles_r0_15(self):
        res = herd_immunity_threshold(15.0)
        assert res["hit"] == pytest.approx(1 - 1 / 15, rel=1e-6)

    def test_r0_2(self):
        res = herd_immunity_threshold(2.0)
        assert res["hit"] == pytest.approx(0.5)

    def test_imperfect_vaccine(self):
        res = herd_immunity_threshold(4.0, vaccine_efficacy=0.9)
        assert res["critical_vaccination_coverage"] == pytest.approx(0.75 / 0.9, rel=1e-6)

    def test_r0_1(self):
        res = herd_immunity_threshold(1.0)
        assert res["hit"] == pytest.approx(0.0)

    def test_r0_negative_raises(self):
        with pytest.raises(ValueError):
            herd_immunity_threshold(-1.0)

    def test_zero_efficacy_raises(self):
        with pytest.raises(ValueError):
            herd_immunity_threshold(3.0, vaccine_efficacy=0.0)

    def test_coverage_capped_at_1(self):
        res = herd_immunity_threshold(10.0, vaccine_efficacy=0.5)
        assert res["critical_vaccination_coverage"] <= 1.0
