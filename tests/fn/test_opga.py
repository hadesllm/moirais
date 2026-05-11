"""Test opga."""
import numpy as np
import pytest
from morie.fn.opga import opga


def test_opga_basic():
    rng = np.random.default_rng(42)
    r = opga(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opga_description():
    rng = np.random.default_rng(42)
    r = opga(n_dims=2, max_iter=50)
    assert r.name
