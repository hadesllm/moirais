"""Tests for heat-wave detection (WMO percentile + consecutive-days)."""

import numpy as np
import pytest

from moirais.fn.heatwv import heat_wave_detect, heatwv


def test_heatwv_detects_single_episode():
    # 100 baseline days at 25, then a 5-day spike well above 90th pct
    T = np.concatenate([
        np.full(100, 25.0),
        np.array([32.0, 34.0, 33.0, 35.0, 32.0]),
        np.full(100, 26.0),
    ])
    r = heatwv(T, percentile=90, min_consecutive_days=3)
    assert r.extra["n_events"] == 1
    assert r.extra["event_lengths"] == [5]
    assert r.extra["event_peaks_C"] == [35.0]
    assert r.value == 5.0


def test_heatwv_rejects_runs_below_min_days():
    # Two 2-day hot spells; with min_consecutive_days=3, zero events
    T = np.concatenate([
        np.full(50, 20.0),
        np.array([30.0, 31.0]),
        np.full(10, 21.0),
        np.array([30.0, 32.0]),
        np.full(50, 22.0),
    ])
    r = heatwv(T, percentile=90, min_consecutive_days=3)
    assert r.extra["n_events"] == 0
    assert r.value == 0.0


def test_heatwv_detects_two_episodes():
    T = np.concatenate([
        np.full(40, 20.0),
        np.full(4, 33.0),
        np.full(20, 22.0),
        np.full(6, 35.0),
        np.full(40, 21.0),
    ])
    r = heatwv(T, percentile=90, min_consecutive_days=3)
    assert r.extra["n_events"] == 2
    assert r.extra["event_lengths"] == [4, 6]
    # Starts at index 40 and 40+4+20 = 64
    assert r.extra["event_start_idx"] == [40, 64]


def test_heatwv_threshold_from_baseline_window():
    # Use only first 100 days as baseline; those are cool, so threshold
    # is low, and later hot stretch crosses it easily.
    T = np.concatenate([
        np.full(100, 20.0),
        np.full(30, 25.0),
    ])
    r = heatwv(T, percentile=90, min_consecutive_days=3,
               baseline_window=(0, 100))
    # Baseline is all 20 → threshold = 20.0 exactly; days > 20 is 30 days
    assert r.extra["threshold_C"] == pytest.approx(20.0)
    assert r.value == 30.0   # 30 hot days, one 30-day event


def test_heatwv_raises_on_nan():
    with pytest.raises(ValueError, match="NaN"):
        heatwv(np.array([25.0, np.nan, 30.0, 32.0, 31.0]))


def test_heatwv_raises_on_too_short_series():
    with pytest.raises(ValueError, match="too short"):
        heatwv(np.array([30.0, 31.0]), min_consecutive_days=3)


def test_heatwv_rejects_percentile_out_of_range():
    T = np.full(100, 20.0)
    with pytest.raises(ValueError, match="percentile must be"):
        heatwv(T, percentile=0)
    with pytest.raises(ValueError, match="percentile must be"):
        heatwv(T, percentile=100)


def test_heatwv_alias_matches():
    assert heatwv is heat_wave_detect


def test_heatwv_hot_day_mask_matches_input_length():
    T = np.array([20.0, 21.0, 30.0, 30.0, 30.0, 21.0])
    r = heatwv(T, percentile=50, min_consecutive_days=3)
    assert len(r.extra["hot_day_mask"]) == len(T)
