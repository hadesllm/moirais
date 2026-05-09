"""Test vmstb."""
import numpy as np
import pytest
from moirais.fn.vmstb import vmstb


def test_vmstb_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vmstb(x=x, y=y, values=v)
    assert isinstance(r.value, float) and np.isfinite(r.value)
    assert r.value >= 0


def test_vmstb_extra():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vmstb(x=x, y=y, values=v)
    assert isinstance(r.name, str) and len(r.name) > 0
    assert len(r.extra["lags"]) == 15
    assert len(r.extra["gamma"]) == 15
    assert all(g >= 0 for g in r.extra["gamma"])
