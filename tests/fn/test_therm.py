"""Tests for morie.fn.therm -- feeling thermometer."""

import numpy as np
from morie.fn.therm import feeling_thermometer_scale, therm


def test_therm_smoke():
    ratings = np.array([0, 50, 100], dtype=float)
    r = therm(ratings)
    assert r.name == "feeling_thermometer_scale"
    assert np.isclose(r.value[0], 0.0)
    assert np.isclose(r.value[1], 0.5)
    assert np.isclose(r.value[2], 1.0)


def test_therm_clip():
    ratings = np.array([-10, 110], dtype=float)
    r = therm(ratings)
    assert np.isclose(r.value[0], 0.0)
    assert np.isclose(r.value[1], 1.0)


def test_therm_alias():
    assert therm is feeling_thermometer_scale
