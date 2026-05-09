"""Tests for Environment Canada Humidex index."""

import numpy as np
import pytest

from moirais.fn.humidex import humidex


def test_humidex_equals_temp_when_dewpoint_is_benchmark():
    # Vapor pressure e(0°C) ≈ 6.11 hPa, so (5/9)(e-10) ≈ -2.16
    # Humidex(T=25, Td=0) = 25 - 2.16 ≈ 22.84
    r = humidex(T_air_C=25.0, T_dew_C=0.0)
    assert r.value == pytest.approx(22.84, abs=0.1)
    assert r.extra["classification"] == "no discomfort"


def test_humidex_environment_canada_reference_value():
    # EC example: T=30°C, Td=25°C → Humidex ≈ 42 ("great discomfort")
    # e(25°C) ≈ 31.7 hPa; H = 30 + (5/9)(31.7 - 10) = 30 + 12.05 ≈ 42
    r = humidex(T_air_C=30.0, T_dew_C=25.0)
    assert r.value == pytest.approx(42.0, abs=0.5)
    assert r.extra["classification"] == "great discomfort"


def test_humidex_classification_bands():
    # Pick T/Td to land in each band
    no_disc = humidex(25, 10)   # mild
    some = humidex(30, 18)      # moderate humid
    great = humidex(33, 25)     # hot humid
    danger = humidex(38, 27)    # severe
    assert no_disc.extra["classification"] == "no discomfort"
    assert some.extra["classification"] in ("some discomfort", "great discomfort")
    assert great.extra["classification"] in ("great discomfort", "dangerous")
    assert danger.extra["classification"] in ("dangerous", "heat stroke imminent")


def test_humidex_rejects_dewpoint_above_air():
    with pytest.raises(ValueError, match="T_dew_C must be"):
        humidex(T_air_C=20.0, T_dew_C=25.0)


def test_humidex_array_input():
    T = np.array([28.0, 32.0, 36.0])
    Td = np.array([18.0, 24.0, 28.0])
    r = humidex(T, Td)
    assert len(r.extra["humidex"]) == 3
    # Monotone increasing with both T and Td — last value should be largest
    assert r.extra["humidex"][-1] > r.extra["humidex"][0]


def test_humidex_vapor_pressure_at_dewpoint_zero():
    r = humidex(T_air_C=20.0, T_dew_C=0.0)
    # At Td=0°C, vapor pressure ≈ 6.11 hPa (reference value)
    assert r.extra["vapor_pressure_hPa"] == pytest.approx(6.11, abs=0.01)
