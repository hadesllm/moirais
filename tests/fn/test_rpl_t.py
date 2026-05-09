"""Tests for rpl_t — placement trend."""
import pandas as pd
from moirais.fn.rpl_t import rplace_trend

def test_rpl_t_basic(otis_df):
    result = rplace_trend(otis_df)
    assert isinstance(result, pd.DataFrame)
    assert "year" in result.columns


def test_cheatsheet():
    from moirais.fn.rpl_t import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
