"""Tests for moirais.fn.pendu."""
import numpy as np
from moirais.fn.pendu import pendulum


def test_pendu_smoke():
    rng = np.random.default_rng(42)
    result = pendulum(theta0=0.3)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.pendu import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
