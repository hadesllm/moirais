"""Test opbat."""
import numpy as np
import pytest
from morie.fn.opbat import opbat


def test_opbat_basic():
    rng = np.random.default_rng(42)
    r = opbat(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opbat_description():
    rng = np.random.default_rng(42)
    r = opbat(n_dims=2, max_iter=50)
    assert r.name
