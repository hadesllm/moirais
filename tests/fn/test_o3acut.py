"""Tests for acute O3 mortality RR (NMMAPS / APHENA)."""

import math

import numpy as np
import pytest

from moirais.fn.o3acut import o3_acute_rr, o3acut


def test_o3acut_reference_gives_rr_one():
    r = o3acut(30.0, reference=30.0)
    assert r.value == pytest.approx(1.0)


def test_o3acut_10ppb_delta_matches_base_rr():
    # At delta = 10 ppb, RR should equal base (1.0052)
    r = o3acut(40.0, reference=30.0, outcome="all_cause")
    assert r.value == pytest.approx(1.0052, abs=1e-4)


def test_o3acut_respiratory_higher_than_all_cause():
    r_all = o3acut(60.0, reference=30.0, outcome="all_cause")
    r_resp = o3acut(60.0, reference=30.0, outcome="respiratory")
    assert r_resp.value > r_all.value


def test_o3acut_ugm3_converts_to_ppb():
    # 60 µg/m³ ≈ 30.6 ppb; delta from ref 0 gives (60/1.96)/10 ≈ 3.06 units
    r_ugm3 = o3acut(60.0, reference=0.0, unit="ug/m3", outcome="all_cause")
    r_ppb = o3acut(60.0 / 1.96, reference=0.0, unit="ppb", outcome="all_cause")
    assert r_ugm3.value == pytest.approx(r_ppb.value, abs=1e-6)


def test_o3acut_ci_brackets_point():
    r = o3acut(50.0, reference=10.0)
    assert r.extra["rr_95lo"] < r.value < r.extra["rr_95hi"]


def test_o3acut_array_input():
    C = np.array([20.0, 30.0, 40.0, 50.0])
    r = o3acut(C, reference=10.0)
    # Monotone increasing
    rrs = r.extra["rr"]
    for i in range(1, len(rrs)):
        assert rrs[i] > rrs[i - 1]


def test_o3acut_unknown_outcome_raises():
    with pytest.raises(KeyError, match="Unknown outcome"):
        o3acut(30.0, outcome="diabetes")


def test_o3acut_rejects_bad_unit():
    with pytest.raises(ValueError, match="unit must be"):
        o3acut(30.0, unit="ppm")


def test_o3acut_outcome_name_normalization():
    # "Cardiovascular" / "cardio-vascular" / "Cardio Vascular" → all valid
    r1 = o3acut(30.0, outcome="cardiovascular")
    r2 = o3acut(30.0, outcome="Cardiovascular")
    assert r1.value == pytest.approx(r2.value)


def test_o3acut_alias_matches():
    assert o3acut is o3_acute_rr
