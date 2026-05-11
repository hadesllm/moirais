"""Tests for rpl_ag — age x gender cross-tab."""
import pandas as pd
from morie.fn.rpl_ag import rplace_age_gender

def test_rpl_ag_basic(otis_df):
    result = rplace_age_gender(otis_df)
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from morie.fn.rpl_ag import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
