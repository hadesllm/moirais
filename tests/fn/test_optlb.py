"""Test optlb."""
import numpy as np
import pytest
from morie.fn.optlb import optlb


def test_optlb_basic():
    rng = np.random.default_rng(42)
    r = optlb(n_dims=2, max_iter=50)
    assert r.value is not None


def test_optlb_description():
    rng = np.random.default_rng(42)
    r = optlb(n_dims=2, max_iter=50)
    assert r.name
