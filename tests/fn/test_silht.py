"""Tests for moirais.fn.silht."""
import numpy as np
from moirais.fn.silht import silht


def test_silht_smoke():
    rng = np.random.default_rng(42)
    n = 30
    X = rng.standard_normal((n, 3))
    labels = rng.integers(0, 3, size=n)
    result = silht(X=X, labels=labels)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.silht import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
