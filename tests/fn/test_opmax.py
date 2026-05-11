"""Test opmax."""
import numpy as np
import pytest
from morie.fn.opmax import opmax


def test_opmax_basic():
    rng = np.random.default_rng(42)
    r = opmax(n_dims=2, max_iter=50)
    assert isinstance(r.value, float)
    assert r.value >= 0, "Minimized sum-of-squares must be non-negative"
    assert np.isfinite(r.value), "Optimization result must be finite"


def test_opmax_description():
    rng = np.random.default_rng(42)
    r = opmax(n_dims=2, max_iter=50)
    assert r.name
