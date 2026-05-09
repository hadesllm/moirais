"""Tests for wet-bulb globe temperature (ISO 7243 heat-stress index)."""

import numpy as np
import pytest

from moirais.fn.wbgt import wbgt, wet_bulb_globe_temp


def test_wbgt_outdoor_known_case():
    # Hot sunny day: Tair 35°C, wet 27°C, globe 42°C
    # WBGT_out = 0.7(27) + 0.2(42) + 0.1(35) = 18.9 + 8.4 + 3.5 = 30.8
    r = wbgt(T_air_C=35.0, T_wet_C=27.0, T_globe_C=42.0, outdoor=True)
    assert r.value == pytest.approx(30.8, rel=1e-6)
    assert r.extra["classification"] == "danger"


def test_wbgt_indoor_uses_different_weights():
    # WBGT_in = 0.7(24) + 0.3(30) = 16.8 + 9.0 = 25.8 — "safe"
    r = wbgt(T_air_C=28.0, T_wet_C=24.0, T_globe_C=30.0, outdoor=False)
    assert r.value == pytest.approx(25.8, rel=1e-6)
    assert r.extra["classification"] == "safe"
    assert r.extra["outdoor"] is False


def test_wbgt_classification_thresholds():
    # Exactly at 28 => safe; at 30.0 => caution; at 32.0 => danger;
    # > 32 => extreme. Use an outdoor solve where we know WBGT maps directly.
    safe = wbgt(T_air_C=28, T_wet_C=28, T_globe_C=28, outdoor=True)
    caution = wbgt(T_air_C=30, T_wet_C=30, T_globe_C=30, outdoor=True)
    danger = wbgt(T_air_C=32, T_wet_C=32, T_globe_C=32, outdoor=True)
    extreme = wbgt(T_air_C=35, T_wet_C=35, T_globe_C=35, outdoor=True)
    assert safe.extra["classification"] == "safe"
    assert caution.extra["classification"] == "caution"
    assert danger.extra["classification"] == "danger"
    assert extreme.extra["classification"] == "extreme"


def test_wbgt_array_input():
    Ta = np.array([30.0, 32.0, 34.0])
    Tw = np.array([25.0, 26.0, 27.0])
    Tg = np.array([35.0, 38.0, 42.0])
    r = wbgt(Ta, Tw, Tg, outdoor=True)
    # Mean should fall in reasonable range
    assert 26.0 < r.value < 32.0
    assert len(r.extra["wbgt_C"]) == 3
    assert len(r.extra["classification"]) == 3


def test_wbgt_shape_mismatch_raises():
    with pytest.raises(ValueError, match="match in shape"):
        wbgt(T_air_C=[30, 32], T_wet_C=[25], T_globe_C=[35])


def test_wbgt_alias_matches():
    assert wbgt is wet_bulb_globe_temp
