"""Tests for moirais.fn.inits -- initial start values."""

import numpy as np
from moirais.fn.inits import initial_start_values, inits


def test_inits_random():
    r = inits(10, n_dims=2, method="random")
    assert r.name == "initial_start_values"
    assert r.value.shape == (10, 2)


def test_inits_grid():
    r = inits(9, n_dims=2, method="grid")
    assert r.value.shape == (9, 2)


def test_inits_alias():
    assert inits is initial_start_values
