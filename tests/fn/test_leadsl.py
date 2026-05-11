"""Tests for lead service-line risk screen."""

import numpy as np
import pytest

from morie.fn.leadsl import lead_service_line_risk, leadsl


def test_leadsl_pre1900_very_high_us():
    r = leadsl(1890, country="us")
    assert r.extra["label"] == "very_high"
    assert r.value == pytest.approx(0.90)


def test_leadsl_1986_ban_year_us():
    # 1986 is the ban year; should fall into low band
    r = leadsl(1986, country="us")
    assert r.extra["label"] == "low"
    assert r.value == pytest.approx(0.05)


def test_leadsl_modern_us_very_low():
    r = leadsl(2020, country="us")
    assert r.extra["label"] == "very_low"
    assert r.value == pytest.approx(0.01)


def test_leadsl_canada_1920_very_high():
    r = leadsl(1920, country="ca")
    assert r.extra["label"] == "very_high"


def test_leadsl_canada_modern_low():
    r = leadsl(2015, country="ca")
    assert r.extra["label"] == "low"


def test_leadsl_array_input_distribution():
    years = np.array([1920, 1950, 1970, 1990, 2015])
    r = leadsl(years, country="us")
    dist = r.extra["band_distribution"]
    # Should have hits in very_high, high, moderate, low
    assert dist.get("very_high", 0) == 1
    assert dist.get("high", 0) == 1
    assert dist.get("moderate", 0) == 1
    assert dist.get("low", 0) == 1 or dist.get("very_low", 0) == 1


def test_leadsl_invalid_country():
    with pytest.raises(ValueError, match="country must be"):
        leadsl(1950, country="uk")


def test_leadsl_country_aliases():
    r1 = leadsl(1950, country="US")
    r2 = leadsl(1950, country="usa")
    r3 = leadsl(1950, country="United States")
    assert r1.extra["label"] == r2.extra["label"] == r3.extra["label"]


def test_leadsl_with_intervals_exposes_bands():
    r = leadsl(1950, country="us", with_intervals=True)
    assert "bands" in r.extra
    assert len(r.extra["bands"]) > 0


def test_leadsl_alias_matches():
    assert leadsl is lead_service_line_risk


def test_leadsl_source_identifies_country():
    r_us = leadsl(1950, country="us")
    r_ca = leadsl(1950, country="ca")
    assert "EPA" in r_us.extra["source"]
    assert "Canada" in r_ca.extra["source"]
