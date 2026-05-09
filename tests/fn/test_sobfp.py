"""Test sobfp."""
import numpy as np
import pytest
from moirais.fn.sobfp import sobfp


def test_sobfp_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = sobfp(x=x, y=y, values=v)
    assert isinstance(r.value, float) and np.isfinite(r.value)
    assert r.value >= 0
    assert r.value == pytest.approx(np.var(v), rel=1e-10)


def test_sobfp_extra():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = sobfp(x=x, y=y, values=v)
    assert isinstance(r.name, str) and len(r.name) > 0
    assert r.extra["n_obs"] == 20
