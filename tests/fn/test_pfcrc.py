"""Tests for moirais.fn.pfcrc -- preference circles."""

import numpy as np
from moirais.fn.pfcrc import preference_circles, pfcrc


def test_pfcrc_smoke():
    r = pfcrc([0, 0], 1.0, n_points=50)
    assert r.name == "preference_circles"
    assert r.value.shape == (50, 2)
    dists = np.sqrt(np.sum(r.value ** 2, axis=1))
    assert np.allclose(dists, 1.0, atol=1e-10)


def test_pfcrc_alias():
    assert pfcrc is preference_circles
