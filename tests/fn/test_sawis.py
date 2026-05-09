"""Test sawis."""
import numpy as np
import pytest
from moirais.fn.sawis import sawis


def test_sawis_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawis(values=vals, n=25)
    assert r.value is not None


def test_sawis_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawis(values=vals, n=25)
    assert r.name
