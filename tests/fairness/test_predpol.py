# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for Phase B — the predictive-policing calibration audit and the
city-agnostic CityProfile data layer.

Known-answer checks plus null/placebo guards: a well-calibrated system
must produce no disparity signal (no false positive), and an injected
group over-prediction must be detected (no false negative).
"""
import numpy as np
import pandas as pd
import pytest

from morie.fairness.cityprofile import (
    CityProfile,
    apply_profile,
    get_city,
    list_cities,
    register_city,
)
from morie.fairness.predpol import (
    predpol_aggregate_areas,
    predpol_calibration_audit,
    predpol_score_disparity,
)


# ── predpol_aggregate_areas ─────────────────────────────────────────

def test_aggregate_basic():
    out = predpol_aggregate_areas(
        area=["a", "a", "b", "b"], risk=[10, 20, 30, 40],
        outcome=[1, 0, 1, 1])
    assert list(out["areas"]) == ["a", "b"]
    assert out["mean_risk"].tolist() == [15.0, 35.0]
    assert out["outcome_rate"].tolist() == [0.5, 1.0]
    assert out["n_records"].tolist() == [2, 2]


def test_aggregate_with_population_gives_per_10k_rate():
    out = predpol_aggregate_areas(
        area=["a", "a", "b", "b"], risk=[10, 20, 30, 40],
        outcome=[1, 0, 1, 1], population={"a": 1000, "b": 2000})
    # a: 1/1000 * 1e4 = 10 ; b: 2/2000 * 1e4 = 10
    assert out["outcome_rate"].tolist() == [10.0, 10.0]


def test_aggregate_majority_group():
    out = predpol_aggregate_areas(
        area=["a", "a", "b", "b"], risk=[1, 2, 3, 4],
        outcome=[0, 0, 0, 0], group=["X", "X", "Y", "Y"])
    assert out["group"].tolist() == ["X", "Y"]


# ── predpol_calibration_audit ───────────────────────────────────────

def test_audit_well_calibrated_no_false_positive():
    # risk ranking == outcome ranking -> every rank gap is zero
    res = predpol_calibration_audit(
        ["d1", "d2", "d3", "d4"], [400, 300, 200, 100],
        [400, 300, 200, 100], ["X", "X", "Y", "Y"])
    assert res.payload["spearman"] == pytest.approx(1.0)
    assert float(res) == pytest.approx(0.0)


def test_audit_perfect_miscalibration():
    res = predpol_calibration_audit(
        ["d1", "d2", "d3", "d4"], [400, 300, 200, 100],
        [100, 200, 300, 400], ["X", "X", "Y", "Y"])
    assert res.payload["spearman"] == pytest.approx(-1.0)


def test_audit_detects_group_over_prediction():
    # X areas: high predicted risk, low realised outcome -> over-predicted
    areas = ["d1", "d2", "d3", "d4", "d5", "d6"]
    risk = [90, 80, 70, 30, 20, 10]
    outcome = [10, 20, 30, 70, 80, 90]
    grp = ["X", "X", "X", "Y", "Y", "Y"]
    res = predpol_calibration_audit(areas, risk, outcome, grp)
    assert res.payload["group_rank_gap"]["X"] == pytest.approx(3.0)
    assert res.payload["group_rank_gap"]["Y"] == pytest.approx(-3.0)
    assert res.payload["spearman"] < 0
    assert float(res) > 0.5  # a real over-prediction signal


def test_audit_requires_two_areas():
    with pytest.raises(ValueError):
        predpol_calibration_audit(["d1"], [1], [1], ["X"])


def test_audit_placebo_random_no_false_alarm():
    # risk independent of outcome, groups random: no group should be
    # flagged as systematically mis-ranked
    rng = np.random.default_rng(11)
    n = 40
    areas = [f"d{i}" for i in range(n)]
    risk = rng.random(n)
    outcome = rng.random(n)
    grp = rng.choice(["X", "Y"], size=n)
    res = predpol_calibration_audit(areas, risk, outcome, grp)
    assert abs(res.payload["spearman"]) < 0.5
    assert abs(float(res)) < 12.0


def test_audit_drops_non_finite_with_warning():
    res = predpol_calibration_audit(
        ["d1", "d2", "d3"], [10.0, 20.0, np.nan], [1.0, 2.0, 3.0],
        ["X", "X", "Y"])
    assert any("non-finite" in w for w in res.warnings)


# ── predpol_score_disparity (descriptive) ───────────────────────────

def test_score_disparity_known_answer():
    score = [9, 10, 11, 19, 20, 21]   # group means 10 and 20
    grp = ["A", "A", "A", "B", "B", "B"]
    res = predpol_score_disparity(score, grp)
    assert float(res) == pytest.approx(10.0)        # mean spread
    assert res.payload["reference"] == "A"          # lowest-scoring
    assert res.payload["gaps"]["B"] == pytest.approx(10.0)
    assert res.payload["significant"] is True       # ANOVA p < 0.05


def test_score_disparity_null_no_difference():
    rng = np.random.default_rng(3)
    grp = np.array(["A"] * 200 + ["B"] * 200)
    score = rng.normal(50, 5, size=400)             # identical dists
    res = predpol_score_disparity(score, grp)
    assert res.payload["significant"] is False
    assert abs(float(res)) < 2.0


def test_score_disparity_requires_two_groups():
    with pytest.raises(ValueError):
        predpol_score_disparity([1, 2, 3], ["A", "A", "A"])


# ── CityProfile ─────────────────────────────────────────────────────

def test_generic_profile_is_identity():
    df = pd.DataFrame({"area": ["a"], "risk": [1.0], "outcome": [0],
                       "population": [100], "group": ["X"]})
    out = apply_profile(df, "generic")
    assert set(out.columns) == {"area", "risk", "outcome",
                                "population", "group"}


def test_register_and_apply_custom_city():
    prof = CityProfile(
        name="testville", area_col="district", risk_col="ssl_score",
        outcome_col="shootings", group_col="race_majority")
    register_city(prof, overwrite=True)
    assert "testville" in list_cities()
    df = pd.DataFrame({"district": ["d1"], "ssl_score": [300.0],
                       "shootings": [2], "race_majority": ["Black"]})
    out = apply_profile(df, "testville")
    assert set(out.columns) == {"area", "risk", "outcome", "group"}
    assert out["risk"].iloc[0] == 300.0


def test_get_city_unknown_raises():
    with pytest.raises(KeyError):
        get_city("atlantis")


def test_register_duplicate_raises_without_overwrite():
    with pytest.raises(ValueError):
        register_city(CityProfile(name="generic", area_col="area"))


def test_apply_profile_missing_column_raises():
    prof = CityProfile(name="x", area_col="missing_col")
    df = pd.DataFrame({"area": ["a"]})
    with pytest.raises(KeyError):
        apply_profile(df, prof)
