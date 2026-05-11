"""Test opnms."""
import numpy as np
import pytest
from morie.fn.opnms import opnms


def test_opnms_basic():
    rng = np.random.default_rng(42)
    r = opnms(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opnms_description():
    rng = np.random.default_rng(42)
    r = opnms(n_dims=2, max_iter=50)
    assert r.name
