"""Tests for WHO 2021 Air Quality Guidelines compliance check."""

import numpy as np
import pytest

from moirais.fn.whoaqg import who_aqg_compliance, whoaqg


def test_whoaqg_pm25_at_or_below_5_is_aqg():
    r = whoaqg(5.0, "pm25", "annual")
    assert r.extra["level"] == "AQG"
    assert r.value == pytest.approx(1.0)


def test_whoaqg_pm25_12_falls_into_it4():
    # PM2.5 annual IT-4 threshold = 10, IT-3 = 15. So 12 -> IT-3.
    r = whoaqg(12.0, "pm25", "annual")
    assert r.extra["level"] == "IT-3"


def test_whoaqg_pm25_50_falls_into_fail_or_it1():
    # IT-1 for PM2.5 annual = 35. So 50 is worse than IT-1 -> fail.
    r = whoaqg(50.0, "pm25", "annual")
    assert r.extra["level"] == "fail"


def test_whoaqg_no2_24h():
    # NO2 24h AQG = 25, IT-2 = 50, IT-1 = 120
    assert whoaqg(20.0, "no2", "24h").extra["level"] == "AQG"
    assert whoaqg(40.0, "no2", "24h").extra["level"] == "IT-2"
    assert whoaqg(150.0, "no2", "24h").extra["level"] == "fail"


def test_whoaqg_ozone_peak_season_accepts_variants():
    # Peak-season averaging: accept "peak_season", "peak-season", "peak season"
    for variant in ("peak_season", "peak-season", "peak season"):
        r = whoaqg(55.0, "o3", variant)
        assert r.extra["level"] == "AQG"


def test_whoaqg_array_compliance_rate():
    # 3 of 5 values meet AQG (≤ 5 µg/m³)
    C = np.array([4.0, 5.0, 8.0, 10.0, 5.0])
    r = whoaqg(C, "pm25", "annual")
    assert r.value == pytest.approx(3 / 5)
    assert r.extra["n_obs"] == 5


def test_whoaqg_unknown_pollutant_raises():
    with pytest.raises(KeyError, match="Unknown pollutant"):
        whoaqg(10.0, "voc", "annual")


def test_whoaqg_unknown_averaging_raises():
    with pytest.raises(KeyError, match="Unknown averaging"):
        whoaqg(10.0, "pm25", "1h")


def test_whoaqg_negative_concentration_raises():
    with pytest.raises(ValueError, match="non-negative"):
        whoaqg(-1.0, "pm25", "annual")


def test_whoaqg_alias_matches():
    assert whoaqg is who_aqg_compliance
