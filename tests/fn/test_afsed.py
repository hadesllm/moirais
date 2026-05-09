"""Test afsed."""
import numpy as np
import pytest
from moirais.fn.afsed import afsed


def test_afsed_basic():
    rng = np.random.default_rng(42)
    yld = rng.uniform(50, 200, 20)
    soil = rng.uniform(0, 1, 20)
    r = afsed(yield_data=yld, soil=soil, n=20)
    assert r.value is not None


def test_afsed_description():
    rng = np.random.default_rng(42)
    yld = rng.uniform(50, 200, 20)
    soil = rng.uniform(0, 1, 20)
    r = afsed(yield_data=yld, soil=soil, n=20)
    assert r.name
