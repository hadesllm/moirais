"""Test opbfg."""
import numpy as np
import pytest
from moirais.fn.opbfg import opbfg


def test_opbfg_basic():
    rng = np.random.default_rng(42)
    r = opbfg(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opbfg_description():
    rng = np.random.default_rng(42)
    r = opbfg(n_dims=2, max_iter=50)
    assert r.name
