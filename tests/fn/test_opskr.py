"""Test opskr."""
import numpy as np
import pytest
from moirais.fn.opskr import opskr


def test_opskr_basic():
    rng = np.random.default_rng(42)
    r = opskr(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opskr_description():
    rng = np.random.default_rng(42)
    r = opskr(n_dims=2, max_iter=50)
    assert r.name
