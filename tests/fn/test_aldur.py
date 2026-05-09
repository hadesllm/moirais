"""Tests for aldur — alert duration."""
from moirais.fn.aldur import aldurn

def test_aldur_basic(otis_df):
    result = aldurn(otis_df)
    assert isinstance(result, (float, dict))


def test_cheatsheet():
    from moirais.fn.aldur import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
