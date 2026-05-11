"""Test opffa."""
import numpy as np
import pytest
from morie.fn.opffa import opffa


def test_opffa_basic():
    rng = np.random.default_rng(42)
    r = opffa(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opffa_description():
    rng = np.random.default_rng(42)
    r = opffa(n_dims=2, max_iter=50)
    assert r.name
