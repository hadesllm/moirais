"""Test opcom."""
import numpy as np
import pytest
from morie.fn.opcom import opcom


def test_opcom_basic():
    rng = np.random.default_rng(42)
    r = opcom(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opcom_description():
    rng = np.random.default_rng(42)
    r = opcom(n_dims=2, max_iter=50)
    assert r.name
