"""Tests for morie.fn.ocmsh -- OC Coombs mesh."""
import numpy as np
from morie.fn.ocmsh import oc_coombs_mesh, ocmsh


def test_alias():
    assert ocmsh is oc_coombs_mesh


def test_smoke():
    normals = np.array([[1.0, 0.0], [0.0, 1.0]])
    cutpoints = np.array([0.0, 0.0])
    r = oc_coombs_mesh(normals, cutpoints, grid_size=10)
    assert r.name == "oc_coombs_mesh"
    assert r.extra["grid_size"] == 10
    assert r.extra["n_bills"] == 2
    assert r.extra["yea_fraction_grid"].shape == (10, 10)
