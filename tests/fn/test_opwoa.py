"""Test opwoa."""
import numpy as np
import pytest
from morie.fn.opwoa import opwoa


def test_opwoa_basic():
    rng = np.random.default_rng(42)
    r = opwoa(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opwoa_description():
    rng = np.random.default_rng(42)
    r = opwoa(n_dims=2, max_iter=50)
    assert r.name
