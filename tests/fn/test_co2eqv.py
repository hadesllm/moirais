"""Tests for CO₂-equivalent conversion (IPCC AR6 GWP100)."""

import pytest

from morie.fn.co2eqv import co2_equivalent, co2eqv


def test_co2eqv_scalar_co2_identity():
    r = co2eqv(1000.0, gas="co2")
    assert r.value == 1000.0
    assert r.extra["gwp100"] == 1.0


def test_co2eqv_scalar_methane_ar6_value():
    # AR6 GWP100 for fossil CH4 = 27.9
    r = co2eqv(100.0, gas="ch4")
    assert r.value == pytest.approx(2790.0)
    assert r.extra["gwp100"] == 27.9


def test_co2eqv_scalar_n2o():
    r = co2eqv(1.0, gas="n2o")
    assert r.value == pytest.approx(273.0)


def test_co2eqv_dict_mixed_inventory():
    # Dairy farm: 1 t CO₂, 25 kg CH₄, 2 kg N₂O
    # 1000 + 25 * 27.9 + 2 * 273 = 1000 + 697.5 + 546 = 2243.5
    r = co2eqv({"co2": 1000, "ch4": 25, "n2o": 2})
    assert r.value == pytest.approx(2243.5, rel=1e-6)
    assert set(r.extra["breakdown_kg"].keys()) == {"co2", "ch4", "n2o"}
    assert r.extra["breakdown_kg"]["co2"] == pytest.approx(1000.0)


def test_co2eqv_case_insensitive_gas_names():
    r1 = co2eqv({"CH4": 10})
    r2 = co2eqv({"ch4": 10})
    r3 = co2eqv({"Ch4": 10})
    assert r1.value == r2.value == r3.value


def test_co2eqv_high_gwp_refrigerant():
    # HFC-23 has GWP100 of 14,600 — one of the strongest per-kg
    r = co2eqv(1.0, gas="hfc-23")
    assert r.value == pytest.approx(14600.0)


def test_co2eqv_unknown_gas_lists_alternatives():
    with pytest.raises(KeyError, match="Available"):
        co2eqv(1.0, gas="cfc-11")  # CFC-11 is not in our table


def test_co2eqv_scalar_requires_gas_arg():
    with pytest.raises(ValueError, match="requires gas="):
        co2eqv(100.0)


def test_co2eqv_alias_matches():
    assert co2eqv is co2_equivalent
