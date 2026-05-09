"""Tests for moirais.fn.apres -- aggregate PRE."""
import numpy as np
from moirais.fn.apres import apre_statistic, apres


def test_alias():
    assert apres is apre_statistic


def test_smoke():
    pre_vals = [0.8, 0.6, 0.9, 0.7]
    r = apre_statistic(pre_vals)
    assert r.name == "apre_statistic"
    assert abs(r.value - 0.75) < 1e-10


def test_extra_fields():
    r = apre_statistic([0.5, 0.5])
    assert "median_pre" in r.extra
    assert "n_roll_calls" in r.extra
    assert r.extra["n_roll_calls"] == 2
