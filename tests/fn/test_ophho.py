"""Test ophho."""
import numpy as np
import pytest
from morie.fn.ophho import ophho


def test_ophho_basic():
    rng = np.random.default_rng(42)
    r = ophho(n_dims=2, max_iter=50)
    assert r.value is not None


def test_ophho_description():
    rng = np.random.default_rng(42)
    r = ophho(n_dims=2, max_iter=50)
    assert r.name
