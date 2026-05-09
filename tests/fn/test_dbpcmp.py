"""Tests for disinfection-byproduct (DBP) drinking-water compliance."""

import pytest

from moirais.fn.dbpcmp import dbp_compliance, dbpcmp


def test_dbpcmp_us_clean_sample():
    r = dbpcmp({"tthm": 40.0, "haa5": 30.0, "bromate": 5.0})
    # All ratios < 1 → compliant
    assert r.extra["compliant"] is True
    assert r.extra["violations"] == []
    # Max ratio is 40/80 = 0.5, 30/60 = 0.5, 5/10 = 0.5 (all tied)
    assert r.value == pytest.approx(0.5)


def test_dbpcmp_us_tthm_violation():
    r = dbpcmp({"tthm": 100.0, "haa5": 40.0})
    assert r.extra["tthm_compliance"] is False
    assert r.extra["haa5_compliance"] is True
    assert r.extra["compliant"] is False
    assert r.value == pytest.approx(100.0 / 80.0)
    assert any("TTHM" in v for v in r.extra["violations"])


def test_dbpcmp_us_multiple_violations():
    r = dbpcmp({"tthm": 100.0, "haa5": 70.0, "bromate": 12.0})
    assert len(r.extra["violations"]) == 3
    # Ratios: TTHMs 100/80=1.25 (max); HAA5 70/60≈1.167; Bromate 12/10=1.2
    assert r.value == pytest.approx(1.25)


def test_dbpcmp_canada_looser_limits():
    # 90 TTHMs would violate US but meet Canada
    r_us = dbpcmp({"tthm": 90.0}, country="us")
    r_ca = dbpcmp({"tthm": 90.0}, country="ca")
    assert r_us.extra["compliant"] is False
    assert r_ca.extra["compliant"] is True


def test_dbpcmp_boundary_exactly_at_mcl():
    # At exactly MCL → ratio = 1.0 → compliant (≤ MCL)
    r = dbpcmp({"tthm": 80.0})
    assert r.extra["compliant"] is True
    assert r.value == pytest.approx(1.0)


def test_dbpcmp_case_insensitive_keys():
    r1 = dbpcmp({"TTHM": 50.0})
    r2 = dbpcmp({"tthm": 50.0})
    assert r1.value == pytest.approx(r2.value)


def test_dbpcmp_unknown_species_raises():
    with pytest.raises(KeyError, match="Unknown DBP"):
        dbpcmp({"tthm": 50.0, "xyz123": 1.0})


def test_dbpcmp_unknown_country_raises():
    with pytest.raises(ValueError, match="country must be"):
        dbpcmp({"tthm": 50.0}, country="uk")


def test_dbpcmp_negative_concentration_raises():
    with pytest.raises(ValueError, match="non-negative"):
        dbpcmp({"tthm": -5.0})


def test_dbpcmp_canada_no_chlorite():
    # Canada table doesn't include chlorite; passing it should error
    with pytest.raises(KeyError, match="Unknown DBP"):
        dbpcmp({"chlorite": 500.0}, country="ca")


def test_dbpcmp_alias_matches():
    assert dbpcmp is dbp_compliance
