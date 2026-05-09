"""Test modlr."""
import pytest
from moirais.fn.modlr import modular_form


def test_modlr_e4():
    r = modular_form(k=4, tau=2.0j)
    assert r.value > 0
    assert r.name == "modular_form"


def test_modlr_invalid_weight():
    with pytest.raises(ValueError):
        modular_form(k=3)
