"""Test opldn."""
import numpy as np
import pytest
from morie.fn.opldn import opldn


def test_opldn_basic():
    rng = np.random.default_rng(42)
    r = opldn(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opldn_description():
    rng = np.random.default_rng(42)
    r = opldn(n_dims=2, max_iter=50)
    assert r.name
