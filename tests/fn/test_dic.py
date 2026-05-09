"""Test dic."""
import pytest
from moirais.fn.dic import deviance_info_criterion


def test_dic_basic():
    r = deviance_info_criterion(D_bar=200.0, D_hat=190.0)
    assert r.name == "dic"
    assert r.extra["p_D"] == 10.0
    assert r.value == 210.0


def test_dic_zero_pd():
    r = deviance_info_criterion(D_bar=100.0, D_hat=100.0)
    assert r.value == 100.0
    assert r.extra["p_D"] == 0.0
