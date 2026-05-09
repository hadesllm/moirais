"""Test agagg."""
import numpy as np
import pytest
from moirais.fn.agagg import agagg


def test_agagg_basic():
    rng = np.random.default_rng(42)
    areas = rng.uniform(10, 500, 15)
    perims = 4 * np.sqrt(areas) * rng.uniform(0.9, 1.3, 15)
    r = agagg(areas=areas, perimeters=perims, n=15)
    assert isinstance(r.value, float)
    assert 0 < r.value <= 1.0, f"Aggregation index {r.value} outside (0, 1]"


def test_agagg_description():
    rng = np.random.default_rng(42)
    areas = rng.uniform(10, 500, 15)
    perims = 4 * np.sqrt(areas) * rng.uniform(0.9, 1.3, 15)
    r = agagg(areas=areas, perimeters=perims, n=15)
    assert r.name
