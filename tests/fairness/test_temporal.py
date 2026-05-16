# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for morie.fairness.temporal — the multi-city temporal audit.

Known-answer checks plus stability/instability guards: a system whose
disparity is constant over time must show no temporal-instability
signal, and a system whose disparity swings must be flagged.
"""
import pytest

from morie.fairness.temporal import predpol_temporal_audit

XY = ["X"] * 5 + ["Y"] * 5


def test_temporal_stable_no_false_instability():
    # identical disparity each period -> DIR temporal range is zero
    period = ["p1"] * 10 + ["p2"] * 10
    city = ["A"] * 20
    pred = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0] * 2
    grp = XY * 2
    res = predpol_temporal_audit(period, city, pred, grp, privileged="X")
    assert res.payload["per_city"]["A"]["dir_range"] == pytest.approx(0.0)
    assert float(res) == pytest.approx(0.0)


def test_temporal_detects_instability():
    # p1: X 1.0 / Y 0.2 -> DIR 0.2 ;  p2: X 1.0 / Y 1.0 -> DIR 1.0
    period = ["p1"] * 10 + ["p2"] * 10
    city = ["A"] * 20
    pred = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0] + [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    grp = XY * 2
    res = predpol_temporal_audit(period, city, pred, grp, privileged="X")
    pc = res.payload["per_city"]["A"]
    assert pc["dir_min"] == pytest.approx(0.2)
    assert pc["dir_max"] == pytest.approx(1.0)
    assert pc["dir_range"] == pytest.approx(0.8)
    assert float(res) == pytest.approx(0.8)
    assert "unstable" in res.interpretation.lower()


def test_temporal_cross_city_divergence():
    # city A: DIR 0.2 both periods ;  city B: DIR 1.0 both periods
    period = (["p1"] * 10 + ["p2"] * 10) * 2
    city = ["A"] * 20 + ["B"] * 20
    pred_a = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0] * 2
    pred_b = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 2
    pred = pred_a + pred_b
    grp = XY * 4
    res = predpol_temporal_audit(period, city, pred, grp, privileged="X")
    assert res.payload["per_city"]["A"]["mean_dir"] == pytest.approx(0.2)
    assert res.payload["per_city"]["B"]["mean_dir"] == pytest.approx(1.0)
    assert res.payload["cross_city_dir_spread"] == pytest.approx(0.8)


def test_temporal_periods_dir_gt1_count():
    # p1: X 1.0 / Y 1.0 -> DIR 1.0 (not >1)
    # p2: X 0.4 / Y 1.0 -> DIR 2.5 (>1)
    period = ["p1"] * 10 + ["p2"] * 10
    city = ["A"] * 20
    pred = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + [1, 1, 0, 0, 0, 1, 1, 1, 1, 1]
    grp = XY * 2
    res = predpol_temporal_audit(period, city, pred, grp, privileged="X")
    assert res.payload["per_city"]["A"]["periods_dir_gt1"] == 1


def test_temporal_skipped_cell_warns():
    # cell A/p1 has only group X -> skipped with a warning
    period = ["p1"] * 5 + ["p2"] * 10
    city = ["A"] * 15
    pred = [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    grp = (["X"] * 5) + XY
    res = predpol_temporal_audit(period, city, pred, grp, privileged="X")
    assert any("skipped" in w for w in res.warnings)


def test_temporal_empty_raises():
    with pytest.raises(ValueError):
        predpol_temporal_audit([], [], [], [])


def test_temporal_misaligned_raises():
    with pytest.raises(ValueError):
        predpol_temporal_audit(["p1"], ["A"], [1], ["X", "Y"])
