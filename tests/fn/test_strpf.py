"""Test strpf."""
from moirais.fn.strpf import string_partition


def test_strpf_basic():
    r = string_partition(tau=0.5j, d=26)
    assert r.value is not None
    assert r.name == "string_partition"


def test_strpf_invalid():
    import pytest
    with pytest.raises(ValueError):
        string_partition(tau=-0.5j)
