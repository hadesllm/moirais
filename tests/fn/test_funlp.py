"""Tests for morie.fn.funlp."""
import numpy as np
from morie.fn.funlp import funlp


def test_funlp_smoke():
    rng = np.random.default_rng(42)
    result = funlp(effects=rng.standard_normal(10), se=rng.uniform(0.1, 0.5, size=10))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.funlp import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
