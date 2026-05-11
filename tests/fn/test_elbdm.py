"""Tests for morie.fn.elbdm -- elbow method MDS dimensionality."""

from morie.fn.elbdm import elbow_mds_dim, elbdm


def test_elbdm_smoke():
    stresses = [0.3, 0.15, 0.08, 0.06, 0.055, 0.054]
    r = elbdm(stresses)
    assert r.name == "elbow_mds_dim"
    assert 1 <= r.value <= 6


def test_elbdm_short():
    r = elbdm([0.5, 0.1])
    assert r.value == 1


def test_elbdm_alias():
    assert elbdm is elbow_mds_dim
