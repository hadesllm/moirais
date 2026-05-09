"""Tests for rpl_gt — gender trend."""
import pandas as pd
from moirais.fn.rpl_gt import rplace_gender_trend

def test_rpl_gt_basic(otis_df):
    result = rplace_gender_trend(otis_df)
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from moirais.fn.rpl_gt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
