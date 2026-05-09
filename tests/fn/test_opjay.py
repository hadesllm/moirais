"""Test opjay."""
import numpy as np
import pytest
from moirais.fn.opjay import opjay


def test_opjay_basic():
    rng = np.random.default_rng(42)
    r = opjay(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opjay_description():
    rng = np.random.default_rng(42)
    r = opjay(n_dims=2, max_iter=50)
    assert r.name
