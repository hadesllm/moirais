"""Tests for moirais.fn.hubbl -- Hubble parameter."""

import numpy as np
import pytest

from moirais.fn.hubbl import hubbl


def test_returns_dict():
    r = hubbl(z=0.0)
    assert isinstance(r, dict)
    for k in ("H_z", "hubble_time_Gyr", "lookback_time_Gyr", "age_Gyr"):
        assert k in r


def test_z_zero():
    r = hubbl(z=0.0, H0=67.4)
    assert r["H_z"] == pytest.approx(67.4, rel=1e-6)
    assert r["lookback_time_Gyr"] == pytest.approx(0.0, abs=1e-10)


def test_age_positive():
    r = hubbl(z=0.0)
    assert r["age_Gyr"] > 10.0
    assert r["age_Gyr"] < 20.0


def test_lookback_increases_with_z():
    r1 = hubbl(z=0.5)
    r2 = hubbl(z=1.0)
    assert r2["lookback_time_Gyr"] > r1["lookback_time_Gyr"]


def test_negative_z_raises():
    with pytest.raises(ValueError):
        hubbl(z=-1.0)
