# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for morie.fairness.simulation — the Noisy-OR detection model
and the synthetic biased-crime-data generator.

The generator's headline guarantee — disparate impact ratio is
approximately ``1 - bias`` — is verified by feeding its output back
through the Phase A audit, which both checks the generator and serves
as an end-to-end audit-pipeline test.
"""
import numpy as np
import pytest

from morie.fairness.metrics import fairness_disparate_impact
from morie.fairness.predpol import predpol_score_disparity
from morie.fairness.simulation import (
    noisy_or_detection,
    simulate_biased_crime_data,
)


# ── noisy_or_detection ──────────────────────────────────────────────

def test_noisy_or_one_officer_in_range():
    res = noisy_or_detection([[0.0, 0.0], [10.0, 10.0]], [[0.1, 0.1]],
                             radius=1.0)
    p = res.payload["probabilities"]
    assert p[0] == pytest.approx(0.85)   # one officer, p=0.85
    assert p[1] == pytest.approx(0.0)    # no officer in range


def test_noisy_or_two_officers_compound():
    res = noisy_or_detection([[0.0, 0.0]], [[0.1, 0.0], [0.0, 0.1]],
                             radius=1.0)
    # 1 - (1 - 0.85)^2 = 1 - 0.0225
    assert res.payload["probabilities"][0] == pytest.approx(0.9775)


def test_noisy_or_no_officers():
    res = noisy_or_detection([[0.0, 0.0]], np.empty((0, 2)), radius=1.0)
    assert res.payload["probabilities"][0] == pytest.approx(0.0)


def test_noisy_or_sampling_is_seeded():
    a = noisy_or_detection([[0.0, 0.0]] * 50, [[0.0, 0.0]], radius=1.0,
                           seed=7)
    b = noisy_or_detection([[0.0, 0.0]] * 50, [[0.0, 0.0]], radius=1.0,
                           seed=7)
    assert list(a.payload["detected"]) == list(b.payload["detected"])


def test_noisy_or_bad_shape_raises():
    with pytest.raises(ValueError):
        noisy_or_detection([0.0, 1.0], [[0.0, 0.0]], radius=1.0)


# ── simulate_biased_crime_data ──────────────────────────────────────

def test_simulator_columns_and_size():
    df = simulate_biased_crime_data(n=500, seed=1)
    assert len(df) == 500
    assert set(df.columns) == {"area", "group", "true_outcome",
                               "detected", "risk_score"}


def test_simulator_bias_zero_is_fair():
    # bias = 0 -> every group flagged at base_rate -> DIR ~ 1
    df = simulate_biased_crime_data(n=8000, bias=0.0, base_rate=0.4, seed=2)
    di = fairness_disparate_impact(df["detected"], df["group"],
                                   privileged="A")
    assert float(di) == pytest.approx(1.0, abs=0.08)
    assert di.payload["adverse_impact"] is False


def test_simulator_injected_bias_is_recovered():
    # bias = 0.6 -> non-reference DIR ~ 1 - 0.6 = 0.4
    df = simulate_biased_crime_data(n=8000, bias=0.6, base_rate=0.4, seed=3)
    di = fairness_disparate_impact(df["detected"], df["group"],
                                   privileged="A")
    assert float(di) == pytest.approx(0.4, abs=0.08)
    assert di.payload["adverse_impact"] is True


def test_simulator_risk_score_shifts_with_bias():
    # bias = 0.6 -> non-reference mean risk score shifted up ~60 points
    df = simulate_biased_crime_data(n=8000, bias=0.6, seed=4)
    res = predpol_score_disparity(df["risk_score"], df["group"])
    assert float(res) == pytest.approx(60.0, abs=12.0)


def test_simulator_areas_are_group_segregated():
    df = simulate_biased_crime_data(n=4000, n_areas=10, seed=5)
    # each area should be dominated by a single group
    for _, sub in df.groupby("area"):
        top_share = sub["group"].value_counts(normalize=True).iloc[0]
        assert top_share == 1.0


def test_simulator_bias_out_of_range_raises():
    with pytest.raises(ValueError):
        simulate_biased_crime_data(bias=1.5)
