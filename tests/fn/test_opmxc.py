"""Test opmxc."""
import numpy as np
import pytest
from moirais.fn.opmxc import opmxc


def test_opmxc_basic():
    rng = np.random.default_rng(42)
    r = opmxc(n_dims=2, max_iter=50)
    assert isinstance(r.value, float)
    assert r.value >= 0, "Minimized sum-of-squares must be non-negative"
    assert np.isfinite(r.value), "Optimization result must be finite"


def test_opmxc_description():
    rng = np.random.default_rng(42)
    r = opmxc(n_dims=2, max_iter=50)
    assert r.name
