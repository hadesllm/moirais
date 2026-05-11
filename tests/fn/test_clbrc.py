"""Test clbrc."""
import numpy as np
import pytest
from morie.fn.clbrc import clbrc


def test_clbrc_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clbrc(data=data, n=30, k=3)
    assert r.value is not None


def test_clbrc_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clbrc(data=data, n=30, k=3)
    assert r.name
