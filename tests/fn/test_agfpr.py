"""Test agfpr."""
import numpy as np
import pytest
from moirais.fn.agfpr import agfpr


def test_agfpr_basic():
    rng = np.random.default_rng(42)
    areas = rng.uniform(10, 500, 15)
    perims = 4 * np.sqrt(areas) * rng.uniform(0.9, 1.3, 15)
    r = agfpr(areas=areas, perimeters=perims, n=15)
    assert r.value is not None


def test_agfpr_description():
    rng = np.random.default_rng(42)
    areas = rng.uniform(10, 500, 15)
    perims = 4 * np.sqrt(areas) * rng.uniform(0.9, 1.3, 15)
    r = agfpr(areas=areas, perimeters=perims, n=15)
    assert r.name
