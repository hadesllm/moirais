"""Tests for indoor radon cancer risk (EPA 2003 / BEIR VI)."""

import numpy as np
import pytest

from moirais.fn.radon import radon, radon_cancer_risk


def test_radon_never_smoker_per_pcil():
    # 1 pCi/L, 70 yrs, never-smoker → 7e-4 lifetime
    r = radon(1.0, smoker=False)
    assert r.value == pytest.approx(7e-4, abs=1e-7)


def test_radon_smoker_per_pcil_ten_times_higher():
    r_s = radon(1.0, smoker=True)
    r_ns = radon(1.0, smoker=False)
    assert r_s.value == pytest.approx(10 * r_ns.value, rel=1e-6)


def test_radon_at_epa_action_level():
    # 4 pCi/L (EPA action), never-smoker → 0.0028 lifetime (0.28%)
    r = radon(4.0, smoker=False)
    assert r.value == pytest.approx(4 * 7e-4, abs=1e-7)
    assert r.extra["above_epa_action_level"] is False   # = action level, not above
    # At 5 pCi/L it would be above
    r_above = radon(5.0, smoker=False)
    assert r_above.extra["above_epa_action_level"] is True


def test_radon_who_reference_lower_than_epa():
    # WHO reference 2.7 pCi/L < EPA 4 pCi/L
    r = radon(3.0, smoker=False)
    assert r.extra["above_who_reference_level"] is True
    assert r.extra["above_epa_action_level"] is False


def test_radon_bqm3_conversion():
    # 1 pCi/L = 37 Bq/m³
    r_pcil = radon(1.0, unit="pCi/L", smoker=False)
    r_bq = radon(37.0, unit="Bq/m3", smoker=False)
    assert r_pcil.value == pytest.approx(r_bq.value, rel=1e-6)
    assert r_bq.extra["radon_pCi_per_L"] == pytest.approx(1.0, abs=1e-6)


def test_radon_exposure_years_scales_linearly():
    r_full = radon(4.0, smoker=False, exposure_years=70)
    r_half = radon(4.0, smoker=False, exposure_years=35)
    assert r_half.value == pytest.approx(r_full.value / 2, rel=1e-6)


def test_radon_array_input():
    R = np.array([1.0, 2.0, 4.0, 10.0])
    r = radon(R, smoker=False)
    risks = r.extra["lifetime_cancer_risk"]
    # Linear in concentration
    assert risks[1] == pytest.approx(2 * risks[0])
    assert risks[2] == pytest.approx(4 * risks[0])
    assert risks[3] == pytest.approx(10 * risks[0])


def test_radon_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        radon(-1.0)


def test_radon_zero_exposure_years_raises():
    with pytest.raises(ValueError, match="exposure_years"):
        radon(4.0, exposure_years=0)


def test_radon_bad_unit_raises():
    with pytest.raises(ValueError, match="unit must be"):
        radon(4.0, unit="sieverts")


def test_radon_alias_matches():
    assert radon is radon_cancer_risk
