"""Test opcnt."""
import numpy as np
import pytest
from moirais.fn.opcnt import opcnt


def test_opcnt_basic():
    rng = np.random.default_rng(42)
    r = opcnt(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opcnt_description():
    rng = np.random.default_rng(42)
    r = opcnt(n_dims=2, max_iter=50)
    assert r.name
