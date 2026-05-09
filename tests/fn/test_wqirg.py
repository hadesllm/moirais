"""Test wqirg."""
import numpy as np
import pytest
from moirais.fn.wqirg import wqirg


def test_wqirg_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqirg(data=data, coords=coords, n=20)
    assert isinstance(r.value, float) and np.isfinite(r.value)
    assert 0 <= r.value <= 14
    assert r.value == pytest.approx(np.mean(data), rel=1e-10)


def test_wqirg_extra():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqirg(data=data, coords=coords, n=20)
    assert isinstance(r.name, str) and len(r.name) > 0
    assert r.extra["n"] == 20
    assert r.extra["std"] >= 0
