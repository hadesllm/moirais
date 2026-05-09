"""Test opmed."""
import numpy as np
import pytest
from moirais.fn.opmed import opmed


def test_opmed_basic():
    rng = np.random.default_rng(42)
    r = opmed(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opmed_description():
    rng = np.random.default_rng(42)
    r = opmed(n_dims=2, max_iter=50)
    assert r.name
