"""Tests for morie.fn.bbgrp — Blackbox group color."""
import numpy as np
from morie.fn.bbgrp import bbgrp


def test_bbgrp_smoke():
    X = np.array([[1, 2], [3, 4], [5, 6.0]])
    r = bbgrp(X, ["D", "R", "D"])
    assert r.name == "bb_group_color"
    assert r.value == 2
    assert r.extra["unique_groups"] == ["D", "R"]


def test_cheatsheet():
    from morie.fn.bbgrp import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
