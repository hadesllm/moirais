"""Tests for rpl_at — age trend."""
import pandas as pd
from moirais.fn.rpl_at import rplace_age_trend

def test_rpl_at_basic(otis_df):
    result = rplace_age_trend(otis_df)
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from moirais.fn.rpl_at import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
