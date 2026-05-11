"""Test opaco."""
import numpy as np
import pytest
from morie.fn.opaco import opaco


def test_opaco_basic():
    rng = np.random.default_rng(42)
    r = opaco(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opaco_description():
    rng = np.random.default_rng(42)
    r = opaco(n_dims=2, max_iter=50)
    assert r.name
