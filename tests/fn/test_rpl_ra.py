"""Tests for rpl_ra — region x age cross-tab."""
import pandas as pd
from morie.fn.rpl_ra import rplace_region_age

def test_rpl_ra_basic(otis_df):
    result = rplace_region_age(otis_df)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] > 0


def test_cheatsheet():
    from morie.fn.rpl_ra import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
