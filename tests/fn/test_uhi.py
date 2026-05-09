"""Tests for Urban Heat Island index."""

import numpy as np
import pytest

from moirais.fn.uhi import uhi, urban_heat_island


def test_uhi_zero_when_urban_equals_rural():
    r = uhi(T_urban_C=20.0, T_rural_C=20.0)
    assert r.value == pytest.approx(0.0)
    assert r.extra["classification"] == "weak"


def test_uhi_positive_urban_warmer():
    r = uhi(T_urban_C=25.0, T_rural_C=21.0)
    assert r.value == pytest.approx(4.0)
    assert r.extra["classification"] == "moderate"


def test_uhi_strong_classification():
    r = uhi(T_urban_C=28.0, T_rural_C=22.0)  # 6°C excess
    assert r.extra["classification"] == "strong"


def test_uhi_array_stats():
    Tu = np.array([22.0, 24.0, 26.0, 28.0, 30.0])
    Tr = np.array([20.0, 21.0, 23.0, 24.0, 25.0])
    r = uhi(Tu, Tr)
    # diffs: 2, 3, 3, 4, 5; mean=3.4, min=2, max=5
    assert r.extra["mean_uhi_C"] == pytest.approx(3.4)
    assert r.extra["min_C"] == pytest.approx(2.0)
    assert r.extra["max_C"] == pytest.approx(5.0)
    assert r.extra["n_obs"] == 5


def test_uhi_timestamps_passed_through():
    import pandas as pd
    ts = pd.date_range("2026-07-01", periods=3, freq="h")
    r = uhi(np.array([25, 26, 28]), np.array([22, 23, 24]), timestamps=ts)
    assert "timestamps" in r.extra
    assert len(r.extra["timestamps"]) == 3


def test_uhi_shape_mismatch_raises():
    with pytest.raises(ValueError, match="match in shape"):
        uhi(T_urban_C=[20, 21], T_rural_C=[19])


def test_uhi_alias_matches():
    assert uhi is urban_heat_island
