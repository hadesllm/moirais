"""Test opass."""
import numpy as np
import pytest
from morie.fn.opass import opass


def test_opass_basic():
    rng = np.random.default_rng(42)
    r = opass(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opass_description():
    rng = np.random.default_rng(42)
    r = opass(n_dims=2, max_iter=50)
    assert r.name
