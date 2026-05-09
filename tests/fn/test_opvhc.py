"""Test opvhc."""
import numpy as np
import pytest
from moirais.fn.opvhc import opvhc


def test_opvhc_basic():
    rng = np.random.default_rng(42)
    r = opvhc(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opvhc_description():
    rng = np.random.default_rng(42)
    r = opvhc(n_dims=2, max_iter=50)
    assert r.name
