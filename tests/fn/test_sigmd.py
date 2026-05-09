"""Tests for moirais.fn.sigmd."""
import numpy as np
from moirais.fn.sigmd import sigmoid


def test_sigmd_smoke():
    rng = np.random.default_rng(42)
    result = sigmoid(x=rng.uniform(40, 45, size=20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.sigmd import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
