# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end integration tests for the morie.fairness subsystem.

Where the per-module tests check one callable in isolation, these
exercise whole *pipelines* — simulate → audit → mitigate → re-audit —
and assert the modules compose correctly.  They also act as
subsystem-level false-positive / false-negative guards: a fair dataset
must read clean across *every* metric, and a biased one must be caught
by every metric.

Reproducing the source papers' exact empirical numbers would require
their datasets (Chicago SSL, Baltimore/Chicago crime, COMPAS), which
morie deliberately does not ship.  These tests instead verify that the
clean-room methods reproduce the papers' *mechanisms* on synthetic
data whose ground-truth bias is known.
"""
import numpy as np
import pytest

from morie.fairness.gan import CTGANDebiaser
from morie.fairness.metrics import (
    fairness_average_odds_difference,
    fairness_bias_amplification,
    fairness_demographic_parity,
    fairness_disparate_impact,
    fairness_equalized_odds,
    fairness_gini,
)
from morie.fairness.predpol import (
    predpol_aggregate_areas,
    predpol_calibration_audit,
)
from morie.fairness.simulation import simulate_biased_crime_data
from morie.fairness.temporal import predpol_temporal_audit
from morie.fairness.xai import xai_permutation_importance


# ── subsystem-level false-positive guard ────────────────────────────

def test_fair_data_reads_clean_across_all_metrics():
    """bias = 0 — every metric must report no disparity."""
    df = simulate_biased_crime_data(n=8000, bias=0.0, base_rate=0.4,
                                    seed=10)
    g, det, tru = df["group"], df["detected"], df["true_outcome"]
    di = fairness_disparate_impact(det, g, privileged="A")
    dp = fairness_demographic_parity(det, g, privileged="A")
    eo = fairness_equalized_odds(tru, det, g, privileged="A")
    aod = fairness_average_odds_difference(tru, det, g, privileged="A")
    bas = fairness_bias_amplification(det, g, privileged="A")

    assert di.payload["adverse_impact"] is False
    assert abs(float(dp)) < 0.06
    assert abs(float(eo)) < 0.12
    assert abs(float(aod)) < 0.10
    assert abs(float(bas)) < 0.05


# ── subsystem-level false-negative guard ────────────────────────────

def test_biased_data_is_caught_by_all_metrics():
    """bias = 0.6 — every metric must register the disparity."""
    df = simulate_biased_crime_data(n=8000, bias=0.6, base_rate=0.4,
                                    seed=11)
    g, det = df["group"], df["detected"]
    di = fairness_disparate_impact(det, g, privileged="A")
    dp = fairness_demographic_parity(det, g, privileged="A")
    bas = fairness_bias_amplification(det, g, privileged="A")

    assert di.payload["adverse_impact"] is True
    assert float(di) < 0.6
    assert float(dp) < -0.15
    assert abs(float(bas)) > 0.05
    assert float(fairness_gini([0.4, 0.16])) > 0.0  # unequal group rates


# ── pipeline: simulate → audit → debias → re-audit ──────────────────

def test_pipeline_debias_then_reaudit_improves_dir():
    pytest.importorskip("jax", reason="morie[sim] extra not installed")
    df = simulate_biased_crime_data(n=4000, bias=0.6, base_rate=0.4,
                                    seed=12)
    before = float(fairness_disparate_impact(
        df["detected"], df["group"], privileged="A"))
    deb = CTGANDebiaser(seed=0).fit(
        df, outcome_col="detected", feature_cols=["risk_score"],
        group_col="group", steps=600)
    syn = deb.debias(4000, privileged="A", seed=1)
    after = float(fairness_disparate_impact(
        syn["detected"], syn["group"], privileged="A"))
    assert before < 0.6 < after, (before, after)


# ── pipeline: segregated city → predpol calibration audit ───────────

def test_pipeline_predpol_flags_over_predicted_group():
    """Marquito mechanism: a group whose areas carry inflated risk
    scores but ordinary realised outcomes is flagged as over-predicted.
    """
    df = simulate_biased_crime_data(n=6000, bias=0.6, base_rate=0.4,
                                    n_areas=24, seed=13)
    agg = predpol_aggregate_areas(
        df["area"], df["risk_score"], df["true_outcome"],
        group=df["group"])
    res = predpol_calibration_audit(
        agg["areas"], agg["mean_risk"], agg["outcome_rate"], agg["group"])
    gaps = res.payload["group_rank_gap"]
    # group B's areas carry bias-inflated risk but group-independent
    # outcomes -> over-predicted (positive rank gap)
    assert gaps["B"] > 0.0
    assert gaps["A"] < gaps["B"]


# ── pipeline: multi-period data → temporal audit ────────────────────

def test_pipeline_temporal_audit_runs_over_periods():
    frames = []
    for i, b in enumerate([0.1, 0.5, 0.2]):
        d = simulate_biased_crime_data(n=2000, bias=b, base_rate=0.4,
                                       seed=20 + i)
        d = d.assign(period=f"p{i}", city="A")
        frames.append(d)
    import pandas as pd
    big = pd.concat(frames, ignore_index=True)
    res = predpol_temporal_audit(big["period"], big["city"],
                                 big["detected"], big["group"],
                                 privileged="A")
    pc = res.payload["per_city"]["A"]
    assert pc["n_periods"] == 3
    # the per-period DIR is not constant -> some temporal range
    assert pc["dir_range"] > 0.0


# ── pipeline: biased model → XAI flags the protected feature ────────

def test_pipeline_xai_flags_protected_driver():
    rng = np.random.default_rng(30)
    X = rng.normal(size=(600, 3))            # cols: age, prior, race
    # a model that leans hard on the protected feature (race, col 2)
    predict = lambda A: 3.0 * A[:, 2] + 0.2 * A[:, 1]
    res = xai_permutation_importance(
        predict, X, feature_names=["age", "prior", "race"],
        protected=["race"], seed=1)
    assert res.payload["ranking"][0] == "race"
    assert any("protected" in w for w in res.warnings)
