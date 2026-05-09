"""Test agalb."""
import numpy as np
import pytest
from moirais.fn.agalb import agalb


def test_agalb_basic():
    rng = np.random.default_rng(42)
    areas = rng.uniform(10, 500, 15)
    perims = 4 * np.sqrt(areas) * rng.uniform(0.9, 1.3, 15)
    r = agalb(areas=areas, perimeters=perims, n=15)
    assert r.value is not None


def test_agalb_description():
    rng = np.random.default_rng(42)
    areas = rng.uniform(10, 500, 15)
    perims = 4 * np.sqrt(areas) * rng.uniform(0.9, 1.3, 15)
    r = agalb(areas=areas, perimeters=perims, n=15)
    assert r.name
