"""Test opazt."""
import numpy as np
import pytest
from morie.fn.opazt import opazt


def test_opazt_basic():
    rng = np.random.default_rng(42)
    r = opazt(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opazt_description():
    rng = np.random.default_rng(42)
    r = opazt(n_dims=2, max_iter=50)
    assert r.name
