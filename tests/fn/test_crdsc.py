"""Tests for moirais.fn.crdsc -- scale coordinates."""

import numpy as np
from moirais.fn.crdsc import scale_coordinates, crdsc


def test_crdsc_smoke():
    V = np.eye(3, 2)
    L = np.array([4.0, 1.0])
    r = crdsc(V, L)
    assert r.name == "scale_coordinates"
    assert r.value.shape == (3, 2)
    assert np.isclose(r.value[0, 0], 2.0)
    assert np.isclose(r.value[1, 1], 1.0)


def test_crdsc_alias():
    assert crdsc is scale_coordinates
