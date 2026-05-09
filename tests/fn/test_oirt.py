"""Tests for ordinal IRT."""
import numpy as np
from moirais.fn.oirt import oirt


def test_oirt_smoke():
    rng = np.random.default_rng(42)
    Y = rng.integers(1, 5, size=(15, 6)).astype(float)
    r = oirt(Y, n_dims=1, n_samples=20, burn_in=10)
    assert r.name == "ordinal_irt_model"
    assert "ideal_points" in r.extra
    assert r.extra["ideal_points"].shape[0] == 15


def test_cheatsheet():
    from moirais.fn.oirt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
