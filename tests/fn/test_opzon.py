"""Test opzon."""
import numpy as np
import pytest
from moirais.fn.opzon import opzon


def test_opzon_basic():
    rng = np.random.default_rng(42)
    r = opzon(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opzon_description():
    rng = np.random.default_rng(42)
    r = opzon(n_dims=2, max_iter=50)
    assert r.name
