"""Tests for moirais.fn.litlw."""
import numpy as np
from moirais.fn.litlw import litlw


def test_litlw_smoke():
    rng = np.random.default_rng(42)
    result = litlw(arrival_rate=0.5, avg_wait=2.0)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.litlw import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
