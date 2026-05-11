"""Test agprm."""
import numpy as np
import pytest
from morie.fn.agprm import agprm


def test_agprm_basic():
    rng = np.random.default_rng(42)
    areas = rng.uniform(10, 500, 15)
    perims = 4 * np.sqrt(areas) * rng.uniform(0.9, 1.3, 15)
    r = agprm(areas=areas, perimeters=perims, n=15)
    assert r.value is not None


def test_agprm_description():
    rng = np.random.default_rng(42)
    areas = rng.uniform(10, 500, 15)
    perims = 4 * np.sqrt(areas) * rng.uniform(0.9, 1.3, 15)
    r = agprm(areas=areas, perimeters=perims, n=15)
    assert r.name
