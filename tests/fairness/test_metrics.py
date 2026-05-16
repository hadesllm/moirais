# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for morie.fairness.metrics — the disparity-metrics core.

Two test families:

* **Known-answer** — hand-computed expected values for each metric.
* **Null / placebo** — bias-free data must yield neutral metrics
  (guards against false positives) and strongly biased data must be
  flagged (guards against false negatives).
"""
import numpy as np
import pytest

from morie.fairness.metrics import (
    fairness_average_odds_difference,
    fairness_bias_amplification,
    fairness_demographic_parity,
    fairness_disparate_impact,
    fairness_equalized_odds,
    fairness_gini,
)

A5B5 = ["A"] * 5 + ["B"] * 5
A4B4 = ["A"] * 4 + ["B"] * 4


# ── disparate impact ────────────────────────────────────────────────

def test_disparate_impact_known_answer():
    # group A: 5/5 favourable; group B: 3/5 -> ratio 0.6
    pred = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    res = fairness_disparate_impact(pred, A5B5, privileged="A")
    assert float(res) == pytest.approx(0.6)
    assert res.payload["adverse_impact"] is True
    assert res.payload["privileged"] == "A"


def test_disparate_impact_null_no_bias():
    # both groups favourable at 0.5 -> ratio 1.0, no adverse impact
    pred = [1, 0, 1, 0, 1, 0, 1, 0]
    res = fairness_disparate_impact(pred, A4B4, privileged="A")
    assert float(res) == pytest.approx(1.0)
    assert res.payload["adverse_impact"] is False


def test_disparate_impact_privileged_inferred_with_warning():
    pred = [1, 1, 1, 1, 1, 0, 0, 0]  # A rate 1.0, B rate 0.0
    res = fairness_disparate_impact(pred, A4B4)
    assert res.payload["privileged"] == "A"
    assert any("inferred" in w for w in res.warnings)


# ── demographic parity ──────────────────────────────────────────────

def test_demographic_parity_known_answer():
    # A rate 0.8, B rate 0.2 -> gap -0.6
    pred = [1, 1, 1, 1, 0, 0, 0, 1, 0, 0]
    res = fairness_demographic_parity(pred, A5B5, privileged="A")
    assert float(res) == pytest.approx(-0.6)


def test_demographic_parity_null():
    pred = [1, 0, 1, 0, 1, 0, 1, 0]
    res = fairness_demographic_parity(pred, A4B4, privileged="A")
    assert float(res) == pytest.approx(0.0)


# ── equalized odds ──────────────────────────────────────────────────

def test_equalized_odds_known_answer():
    # A: TPR 1.0 / FPR 0.0 ;  B: TPR 0.5 / FPR 1.0
    truth = [1, 0, 1, 0, 1, 0, 1, 0]
    pred = [1, 0, 1, 0, 1, 1, 0, 1]
    res = fairness_equalized_odds(truth, pred, A4B4, privileged="A")
    assert res.payload["tpr_gaps"]["B"] == pytest.approx(-0.5)
    assert res.payload["fpr_gaps"]["B"] == pytest.approx(1.0)
    assert float(res) == pytest.approx(1.0)  # worst gap by magnitude
    assert res.payload["violation"] is True


def test_equalized_odds_null():
    truth = [1, 0, 1, 0, 1, 0, 1, 0]
    pred = [1, 0, 1, 0, 1, 0, 1, 0]  # identical error profile
    res = fairness_equalized_odds(truth, pred, A4B4, privileged="A")
    assert float(res) == pytest.approx(0.0)
    assert res.payload["violation"] is False


# ── average odds difference ─────────────────────────────────────────

def test_average_odds_difference_known_answer():
    # AOD_B = 0.5 * ((FPR_B-FPR_A) + (TPR_B-TPR_A))
    #       = 0.5 * ((1.0-0.0) + (0.5-1.0)) = 0.25
    truth = [1, 0, 1, 0, 1, 0, 1, 0]
    pred = [1, 0, 1, 0, 1, 1, 0, 1]
    res = fairness_average_odds_difference(truth, pred, A4B4, privileged="A")
    assert float(res) == pytest.approx(0.25)


# ── Gini ────────────────────────────────────────────────────────────

def test_gini_perfect_equality():
    assert float(fairness_gini([5, 5, 5, 5])) == pytest.approx(0.0)


def test_gini_concentration():
    assert float(fairness_gini([0, 0, 0, 100])) == pytest.approx(0.75)


def test_gini_per_group():
    res = fairness_gini([1, 1, 1, 1, 0, 0, 0, 40], group=A4B4)
    assert res.payload["per_group"]["A"] == pytest.approx(0.0)
    assert res.payload["per_group"]["B"] > 0.5


# ── bias amplification score (arXiv:2603.18987) ─────────────────────

def test_bias_amplification_known_answer():
    # A rate 1.0, B rate 0.0 -> Δ_parity = -1.0 ; Gini([1,0]) = 0.5
    # BAS = -1.0 * 0.5 = -0.5
    pred = [1, 1, 1, 1, 0, 0, 0, 0]
    res = fairness_bias_amplification(pred, A4B4, privileged="A")
    assert res.payload["demographic_parity_gap"] == pytest.approx(-1.0)
    assert res.payload["gini"] == pytest.approx(0.5)
    assert float(res) == pytest.approx(-0.5)


def test_bias_amplification_null():
    # both groups at 0.5 -> Δ_parity 0, Gini 0 -> BAS 0
    pred = [1, 0, 1, 0, 1, 0, 1, 0]
    res = fairness_bias_amplification(pred, A4B4, privileged="A")
    assert float(res) == pytest.approx(0.0)


# ── error handling ──────────────────────────────────────────────────

def test_requires_two_groups():
    with pytest.raises(ValueError):
        fairness_disparate_impact([1, 0, 1], ["A", "A", "A"])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        fairness_disparate_impact([1, 0, 1], ["A", "B"])


def test_unknown_privileged_raises():
    with pytest.raises(ValueError):
        fairness_disparate_impact([1, 0, 1, 0], ["A", "A", "B", "B"], privileged="Z")


# ── null / placebo: false-positive guard ────────────────────────────

def test_placebo_no_false_positive():
    """Labels independent of group: every metric must read near-neutral."""
    rng = np.random.default_rng(42)
    n = 4000
    group = rng.choice(["A", "B"], size=n)
    pred = rng.integers(0, 2, size=n)
    truth = rng.integers(0, 2, size=n)

    di = fairness_disparate_impact(pred, group, privileged="A")
    dp = fairness_demographic_parity(pred, group, privileged="A")
    eo = fairness_equalized_odds(truth, pred, group, privileged="A")

    assert abs(float(dp)) < 0.06, "demographic-parity false positive"
    assert 0.85 < float(di) < 1.20, "disparate-impact false positive"
    assert abs(float(eo)) < 0.15, "equalized-odds false positive"
    assert di.payload["adverse_impact"] is False


# ── injected bias: false-negative guard ─────────────────────────────

def test_injected_bias_is_detected():
    """Group B systematically denied: the audit MUST flag it."""
    rng = np.random.default_rng(7)
    half = 1000
    group = np.array(["A"] * half + ["B"] * half)
    favoured_prob = np.where(group == "A", 0.80, 0.20)
    pred = (rng.random(2 * half) < favoured_prob).astype(int)

    di = fairness_disparate_impact(pred, group, privileged="A")
    dp = fairness_demographic_parity(pred, group, privileged="A")

    assert di.payload["adverse_impact"] is True, "missed real bias"
    assert float(di) < 0.8
    assert float(dp) < -0.4


# ── RichResult contract ─────────────────────────────────────────────

def test_results_are_rich_and_dict_like():
    res = fairness_disparate_impact([1, 1, 0, 0], ["A", "A", "B", "B"], privileged="A")
    assert isinstance(res, dict)            # legacy dict callers keep working
    assert "value" in res.payload
    assert res.title and res.interpretation  # paragraph-level output
    assert "Disparate Impact" in res.summary()
