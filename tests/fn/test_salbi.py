"""Test salbi."""
import numpy as np
import pytest
from morie.fn.salbi import salbi


def test_salbi_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = salbi(values=vals, n=25)
    assert r.value is not None


def test_salbi_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = salbi(values=vals, n=25)
    assert r.name
