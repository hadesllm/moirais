"""Tests for Environment Canada Wind Chill index."""

import numpy as np
import pytest

from morie.fn.wndchl import wind_chill, wndchl


def test_wndchl_environment_canada_reference():
    # Recomputed manually:
    # T=-10, v=20, v^0.16 = 20^0.16 ≈ 1.6154
    # W = 13.12 + 0.6215(-10) - 11.37(1.6154) + 0.3965(-10)(1.6154)
    # W = 13.12 - 6.215 - 18.367 - 6.405 ≈ -17.87
    # EC publishes this as "-18" (rounded to whole °C).
    r = wndchl(-10.0, 20.0)
    assert r.value == pytest.approx(-17.87, abs=0.3)


def test_wndchl_classification_ranges():
    low = wndchl(-5.0, 10.0)          # mild, should be > -28
    # T=-15, v=25 → W ≈ -25 (low risk boundary)
    mod = wndchl(-20.0, 25.0)         # approaches the -28 boundary
    # T=-30, v=40 → W ≈ -47.5 (10-min range)
    risk10 = wndchl(-30.0, 40.0)
    # T=-40, v=50 → W ≈ -60 (<-55 = < 2 min)
    risk_extreme = wndchl(-40.0, 50.0)

    assert low.extra["classification"] == "low risk"
    assert mod.extra["classification"] in ("low risk", "frostbite 30 min")
    assert risk10.extra["classification"] == "frostbite 10 min"
    assert risk_extreme.extra["classification"] in ("frostbite 5 min",
                                                     "frostbite < 2 min")


def test_wndchl_outside_envelope_returns_T():
    # Warm day (T > 10°C) or dead calm (< 4.8 km/h) → formula
    # returns air temp unchanged.
    r_warm = wndchl(15.0, 20.0)
    assert r_warm.value == pytest.approx(15.0)
    assert r_warm.extra["in_envelope"] is False

    r_calm = wndchl(-5.0, 2.0)
    assert r_calm.value == pytest.approx(-5.0)


def test_wndchl_array_input():
    T = np.array([-5.0, -10.0, -20.0, -30.0])
    V = np.array([10.0, 20.0, 30.0, 40.0])
    r = wndchl(T, V)
    assert len(r.extra["wind_chill_C"]) == 4
    # Monotone: colder + windier → lower wind chill
    assert r.extra["wind_chill_C"][3] < r.extra["wind_chill_C"][0]


def test_wndchl_negative_wind_raises():
    with pytest.raises(ValueError, match="non-negative"):
        wndchl(-10.0, -5.0)


def test_wndchl_alias_matches():
    assert wndchl is wind_chill
