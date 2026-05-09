"""Test opagg."""
import numpy as np
import pytest
from moirais.fn.opagg import opagg


def test_opagg_basic():
    rng = np.random.default_rng(42)
    r = opagg(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opagg_description():
    rng = np.random.default_rng(42)
    r = opagg(n_dims=2, max_iter=50)
    assert r.name
