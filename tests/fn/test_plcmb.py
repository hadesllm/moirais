"""Tests for moirais.fn.plcmb -- Coombs mesh visualization."""
import numpy as np
from moirais.fn.plcmb import plot_coombs_data, plcmb


def test_alias():
    assert plcmb is plot_coombs_data


def test_smoke():
    mesh = {
        "yea_fraction_grid": np.random.default_rng(42).uniform(size=(10, 10)),
        "grid_x": np.linspace(-1, 1, 10),
        "grid_y": np.linspace(-1, 1, 10),
    }
    r = plot_coombs_data(mesh)
    assert r.name == "plot_coombs_data"
    assert "grid" in r.extra
    assert r.extra["shape"] == [10, 10]
