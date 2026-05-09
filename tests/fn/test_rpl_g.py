"""Tests for rpl_g — placement by gender."""
import pandas as pd
from moirais.fn.rpl_g import rplace_by_gender

def test_rpl_g_basic(otis_df):
    result = rplace_by_gender(otis_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_cheatsheet():
    from moirais.fn.rpl_g import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
