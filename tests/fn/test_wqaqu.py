"""Test wqaqu."""
import numpy as np
import pytest
from moirais.fn.wqaqu import wqaqu


def test_wqaqu_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqaqu(data=data, coords=coords, n=20)
    assert r.value is not None


def test_wqaqu_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqaqu(data=data, coords=coords, n=20)
    assert r.name
