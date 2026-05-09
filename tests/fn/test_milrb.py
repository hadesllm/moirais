"""Tests for moirais.fn.milrb."""
import numpy as np
from moirais.fn.milrb import milrb


def test_milrb_smoke():
    rng = np.random.default_rng(42)
    result = milrb(n=5)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.milrb import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
