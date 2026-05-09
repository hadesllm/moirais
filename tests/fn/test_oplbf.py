"""Test oplbf."""
import numpy as np
import pytest
from moirais.fn.oplbf import oplbf


def test_oplbf_basic():
    rng = np.random.default_rng(42)
    r = oplbf(n_dims=2, max_iter=50)
    assert r.value is not None


def test_oplbf_description():
    rng = np.random.default_rng(42)
    r = oplbf(n_dims=2, max_iter=50)
    assert r.name
