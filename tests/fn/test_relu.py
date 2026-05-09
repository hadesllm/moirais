"""Tests for moirais.fn.relu."""
import numpy as np
from moirais.fn.relu import relu


def test_relu_smoke():
    rng = np.random.default_rng(42)
    result = relu(x=rng.uniform(40, 45, size=20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.relu import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
