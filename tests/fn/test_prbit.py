"""Tests for moirais.fn.prbit — ordinal probit coefficients."""
import numpy as np
from moirais.fn.prbit import prbit


def test_prbit_smoke():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 2))
    Y = (X @ [1.0, 0.5] + rng.normal(0, 0.3, 50))
    r = prbit(Y, X)
    assert r.name == "ordinal_probit_coefficients"
    assert "coefficients" in r.extra
    assert len(r.extra["coefficients"]) == 2


def test_cheatsheet():
    from moirais.fn.prbit import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
