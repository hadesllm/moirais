"""Tests for rpl_a — placement by age."""
import pandas as pd
from moirais.fn.rpl_a import rplace_by_age

def test_rpl_a_basic(otis_df):
    result = rplace_by_age(otis_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_cheatsheet():
    from moirais.fn.rpl_a import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
