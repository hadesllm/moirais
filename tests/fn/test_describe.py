"""Tests for morie.fn.describe — the pedagogical guide loader."""

from __future__ import annotations

import pytest

from morie.fn import describe


def test_describe_loads_full_md_for_welcht():
    """welcht has a hand-authored describe_welcht.md — should produce
    a full multi-section RichResult, no skeleton warning."""
    r = describe("welcht")
    assert "Welch" in r.title
    assert not r.warnings  # no skeleton warning
    # All 9 sections should be parsed
    section_titles = {s["title"] for s in r.sections}
    expected = {
        "WHAT IT DOES", "WHEN TO USE", "WHEN NOT TO USE",
        "ASSUMPTIONS", "FORMULA", "INPUTS / OUTPUTS",
        "WORKED EXAMPLE", "COMMON MISTAKES", "REFERENCES",
    }
    assert expected.issubset(section_titles), (
        f"missing sections: {expected - section_titles}"
    )


def test_describe_full_text_contains_section_headers():
    r = describe("welcht")
    text = str(r)
    assert "WHAT IT DOES" in text
    assert "WHEN TO USE" in text
    assert "ASSUMPTIONS" in text
    assert "FORMULA" in text
    assert "WORKED EXAMPLE" in text
    assert "COMMON MISTAKES" in text
    assert "REFERENCES" in text


def test_describe_skeleton_for_callable_without_md():
    """A callable that doesn't have a describe_*.md file gets the
    auto-generated skeleton with a warning."""
    r = describe("xbar")
    assert r.warnings
    assert any("skeleton" in w.lower() or "describe_" in w for w in r.warnings)


def test_describe_unknown_callable_raises():
    with pytest.raises(ValueError) as excinfo:
        describe("totally_made_up_function")
    assert "unknown callable" in str(excinfo.value).lower()


def test_describe_case_insensitive_lookup():
    r1 = describe("welcht")
    r2 = describe("WELCHT")
    assert str(r1) == str(r2)


def test_describe_first_batch_authored():
    """Authored describe_*.md callables — all should have non-skeleton
    output with all 9 standard sections."""
    authored = (
        # First batch (round 1)
        "welcht", "paired", "manwhi", "kwallis", "shapir", "cohend",
        # Second batch (round 2)
        "wilcoxn", "kentau", "spearm", "fishex", "mcnem", "aurroc",
        "kmsurv", "logrnk",
        # Third batch (round 3) — descriptive stats + GoF + info criteria
        "covar", "skew", "kurt", "iqrng", "mad", "akike", "bayic",
        "mcfadr", "ksonebs", "anddrl",
        # Fourth batch (round 4) — models + multivariate
        "rdgr", "lasr", "elnetr", "pcaprx", "kmeans2", "mlenrm", "mlepoi",
        "mahalan", "hotelt2", "glmpoi",
        # Fifth batch (round 5) — multi-test + Bayesian + info + power + distance
        "bonfer", "holm", "bhfdr", "priorbt", "bayesf", "hpdint",
        "kldivg", "tventr", "powtt2", "npowtt", "wasdst", "diffd",
        # Sixth batch (round 6) — effect sizes + categorical + spatial + calibration
        "cohensh", "hedgeg", "glassd", "etasq", "omeg2", "tschpr",
        "morani", "gearyc", "brierl", "logloss", "mcc", "yindex",
        # Seventh batch (round 7) — model-fit + econometrics + remaining
        "rsq", "lrtst", "kentau", "spearm", "somerd", "gkgam", "dwtest",
        "ljbox", "bptest", "waldjt", "wald",
    )
    for name in authored:
        r = describe(name)
        assert not r.warnings, f"{name}: expected full md, got skeleton"
        section_titles = {s["title"] for s in r.sections}
        assert "WHAT IT DOES" in section_titles, f"{name}: missing WHAT IT DOES"


def test_describe_payload_access():
    r = describe("welcht")
    assert r.payload["name"] == "welcht"
    assert r.payload["category"]
    assert r.payload["sections"]
