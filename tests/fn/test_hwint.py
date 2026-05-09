"""Tests for moirais.fn.hwint."""
import numpy as np
from moirais.fn.hwint import hwint


def test_hwint_smoke():
    rng = np.random.default_rng(42)
    y = np.sin(np.linspace(0, 4*np.pi, 48)) + rng.normal(0, 0.1, 48)
    result = hwint(y=y, season=12)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.hwint import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
