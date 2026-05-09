"""Test dlasm."""
import numpy as np
import pytest
from moirais.fn.dlasm import dlasm


def test_dlasm_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = dlasm(points=pts, n=40)
    assert r.value is not None


def test_dlasm_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = dlasm(points=pts, n=40)
    assert r.name
