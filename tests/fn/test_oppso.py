"""Test oppso."""
import numpy as np
import pytest
from morie.fn.oppso import oppso


def test_oppso_basic():
    rng = np.random.default_rng(42)
    r = oppso(n_dims=2, max_iter=50)
    assert r.value is not None


def test_oppso_description():
    rng = np.random.default_rng(42)
    r = oppso(n_dims=2, max_iter=50)
    assert r.name
