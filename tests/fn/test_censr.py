"""Tests for moirais.fn.censr."""
import numpy as np
from moirais.fn.censr import censr


def test_censr_smoke():
    rng = np.random.default_rng(42)
    result = censr(y=rng.uniform(-80, -75, size=20), x=rng.uniform(40, 45, size=20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.censr import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
