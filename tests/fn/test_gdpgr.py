"""Tests for moirais.fn.gdpgr."""
import numpy as np
from moirais.fn.gdpgr import gdp_growth


def test_gdpgr_smoke():
    rng = np.random.default_rng(42)
    result = gdp_growth(gdp_series=rng.uniform(10, 100, size=50))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.gdpgr import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
