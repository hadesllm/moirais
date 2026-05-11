"""Tests for morie.cheatsheet — enriched per-fn help cards."""
from __future__ import annotations

from morie import cheatsheet as cs


def test_returns_multiline_block_for_known_fn():
    out = cs.cheatsheet("huberw")
    # Header banner with the fn name.
    assert "huberw" in out
    # When-to-use section is always present.
    assert "When to use this" in out
    # And ends with a quote (either from the Robust pool or _default).
    assert '"' in out


def test_unknown_fn_does_not_crash():
    out = cs.cheatsheet("definitely_not_a_real_fn_name_zzz")
    assert "not found" in out


def test_quote_is_deterministic_for_same_fn():
    a = cs.cheatsheet("icc1")
    b = cs.cheatsheet("icc1")
    assert a == b, "quote pick must be stable for the same fn"


def test_different_fns_can_pick_different_quotes():
    # Doesn't have to be the case — but with prefix-keyed pools it
    # would be a bug if every fn in the same prefix got the same line
    # and there was more than one option to pick from.
    quotes = []
    for fn in ("icc1", "icc2", "icc3", "blupr", "remlfn", "vcomp"):
        out = cs.cheatsheet(fn)
        last_line = out.strip().splitlines()[-1]
        quotes.append(last_line)
    # At least some variety — duplication of every quote signals the
    # picker is broken.
    assert len(set(quotes)) >= 2


def test_when_to_use_present_for_known_categories():
    # Robust category should produce robust-specific guidance, not
    # the _default placeholder.
    out = cs.cheatsheet("huberw")
    # The robust guidance contains the word "outlier" or "Huber".
    assert "outlier" in out.lower() or "M-estimat" in out


def test_category_quotes_have_default_pool():
    assert "_default" in cs.CATEGORY_QUOTES
    assert len(cs.CATEGORY_QUOTES["_default"]) >= 1


def test_when_to_use_has_default_guidance():
    assert "_default" in cs.WHEN_TO_USE
