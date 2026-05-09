"""Tests for moirais.fn.idcon — ideological constraint."""
import numpy as np
from moirais.fn.idcon import idcon


def test_idcon_smoke():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 3))
    r = idcon(X)
    assert r.name == "ideological_constraint"
    assert 0 <= r.value <= 1
    assert r.extra["n_issues"] == 3


def test_cheatsheet():
    from moirais.fn.idcon import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
