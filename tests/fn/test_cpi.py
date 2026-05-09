"""Tests for moirais.fn.cpi."""
import numpy as np
import pytest
from moirais.fn.cpi import cpi_inflation


def test_cpi_no_change():
    prices = np.array([10.0, 20.0, 30.0])
    result = cpi_inflation(prices_base=prices, prices_current=prices)
    assert isinstance(result.value, float) and np.isfinite(result.value)
    assert result.value == pytest.approx(100.0, rel=1e-10)
    assert result.extra["inflation_rate"] == pytest.approx(0.0, abs=1e-10)


def test_cpi_doubled_prices():
    base = np.array([10.0, 20.0, 30.0])
    current = base * 2
    result = cpi_inflation(prices_base=base, prices_current=current)
    assert result.value == pytest.approx(200.0, rel=1e-10)
    assert result.extra["inflation_rate"] == pytest.approx(100.0, rel=1e-10)


def test_cpi_positive():
    rng = np.random.default_rng(42)
    result = cpi_inflation(
        prices_base=rng.uniform(10, 100, size=30),
        prices_current=rng.uniform(10, 100, size=30),
    )
    assert isinstance(result.value, float) and np.isfinite(result.value)
    assert result.value > 0
    assert isinstance(result.name, str) and len(result.name) > 0
    assert result.extra["n_items"] == 30


def test_cheatsheet():
    from moirais.fn.cpi import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
