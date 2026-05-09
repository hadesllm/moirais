"""Test opsca."""
import numpy as np
import pytest
from moirais.fn.opsca import opsca


def test_opsca_basic():
    rng = np.random.default_rng(42)
    r = opsca(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opsca_description():
    rng = np.random.default_rng(42)
    r = opsca(n_dims=2, max_iter=50)
    assert r.name
