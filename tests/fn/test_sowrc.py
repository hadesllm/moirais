"""Test sowrc."""
import numpy as np
import pytest
from morie.fn.sowrc import sowrc


def test_sowrc_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = sowrc(data=data, depth=depth, n=20)
    assert r.value is not None


def test_sowrc_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = sowrc(data=data, depth=depth, n=20)
    assert r.name
