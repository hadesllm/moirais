"""Enriched cheatsheets for fn/ entries.

the author's brief: "make this fully descriptive, and assume our users know
nothing about anything (just like how I did a few years ago due to
the wrong teachers)."

Plain `cheatsheet()` on each fn returns one terse line (e.g.
"huberw: Huber psi-weight function"). That is fine for tab-complete
hints but is useless to someone who has never met an M-estimator
before. This module produces a multi-section help block with:

  1. Short description (from the fn's existing one-line cheatsheet)
  2. When to use this -- plain-English category guidance
  3. A category-themed quote, because learning sticks better when
     it's threaded through stories the reader already knows
  4. Reference (paper / book the formula comes from, when known)
  5. Module path (so the reader can `from morie.fn import …`)

Public API: `cheatsheet(fn_name) -> str`. Used by:
  - the TUI Help screen ('h' key) when the user expands a fn
  - `morie cheatsheet <fn>` CLI subcommand
  - the documentation generator (Sphinx pre-build hook)
"""
from __future__ import annotations

import importlib
import inspect
import json
import re
from pathlib import Path
from typing import Iterable

# ── category quotes ──────────────────────────────────────────────────────────
# Mapping from category-PREFIX to a themed line. Prefix matching means
# "MultilevelICC" / "MultilevelR2" / "MultilevelRandom" all share one quote
# pool. New batch categories pick up matching prefixes for free.
#
# Quotes are deliberately drawn from the same fictional universes the fn/
# tree's own short names already gravitate to: Star Wars, Marvel, DC,
# Matrix, Lord of the Rings, plus the occasional non-genre line that
# fits the math.

CATEGORY_QUOTES: dict[str, list[str]] = {
    "Multilevel": [
        "The unexamined life is not worth living. -- Socrates",
        "Knowing yourself is true wisdom. -- Lao Tzu (clusters know themselves; that is what variance components measure).",
    ],
    "Robust": [
        "Patience is bitter, but its fruit is sweet. -- Aristotle (M-estimation, in one line).",
        "We suffer more often in imagination than in reality. -- Seneca (and your outliers are usually that).",
    ],
    "DL": [
        "What is now proved was once only imagined. -- William Blake (back-propagation included).",
        "Out of chaos comes order. -- Heraclitus (also: Adam optimiser).",
    ],
    "IRT": [
        "Real knowledge is to know the extent of one's ignorance. -- Confucius (so does an information curve).",
    ],
    "Copula": [
        "All things change to something new, and that new is itself changing. -- Marcus Aurelius (the dependence structure, in one line).",
    ],
    "Network": [
        "The whole is greater than the sum of its parts. -- Aristotle (community detection, in one line).",
    ],
    "Causal": [
        "Time discovers truth. -- Seneca (so does a longer panel).",
    ],
    "Bayesian": [
        "Knowledge itself is power. -- Francis Bacon (and your prior is part of it).",
    ],
    "Spatial": [
        "Everything flows. -- Heraclitus (and so does spatial autocorrelation).",
    ],
    "TimeSeries": [
        "We can know more than we can tell. -- Polanyi (an AR(p) is mostly what we cannot tell).",
    ],
    "CausalDID": [
        "There are no two-way fixed effects, only the choices we make about what to compare to what.",
    ],
    "CausalSyntheticControl": [
        "Build the counterfactual that the data refused to give you. -- paraphrased Abadie.",
    ],
    "CausalRDD": [
        "This far, no further. -- the regression discontinuity, in one line.",
    ],
    "CausalML": [
        "He who has a why to live can bear almost any how. -- Nietzsche (also: cross-fitting).",
    ],
    "CausalSensitivity": [
        "Trust, but verify. -- proverb (and e-values).",
    ],
    "Information": [
        "To know the answer, you must first know what surprises you. -- after Shannon.",
    ],
    "Differential": [
        "Privacy is the calibrated noise that lets the truth still be heard.",
    ],
    "Survey": [
        "Do not weight what you cannot measure. -- every survey methodologist, paraphrased.",
    ],
    "_default": [
        "Patience and perseverance have a magical effect. -- John Quincy Adams.",
        "What we cannot speak about we must pass over in silence. -- Wittgenstein.",
    ],
}


# ── educational guidance per category prefix ─────────────────────────────────
WHEN_TO_USE: dict[str, str] = {
    "Multilevel": (
        "Use when your data has a hierarchy (students inside schools, "
        "patients inside hospitals, repeated measures inside subjects).  "
        "A plain regression treats those as independent rows, but they "
        "aren't -- kids in the same school share a teacher, a building, "
        "and a hundred other invisible variables.  Multilevel models "
        "let the cluster have its own intercept (and sometimes its own "
        "slope), so your standard errors aren't lying to you."
    ),
    "Robust": (
        "Use when one or two extreme observations would otherwise "
        "dominate your estimate.  OLS minimises *squared* error, which "
        "means a single row 100 standard deviations from the mean has "
        "10000x the pull of a typical row.  Huber/MM/LTS replace that "
        "with bounded influence functions -- outliers stop yanking the "
        "fit around but still show up in the residuals so you notice."
    ),
    "DL": (
        "Use when the relationship is nonlinear and you have enough "
        "data + compute to learn the shape from scratch.  Neural-net "
        "primitives (attention, RoPE, RMSNorm, MoE) are how modern "
        "LLMs and vision models work end-to-end; ship one of these "
        "into a pipeline only when a simpler model has actually "
        "failed at the task -- not because GPT-4 sounds cool."
    ),
    "IRT": (
        "Use when you're scoring a TEST and 'right answers' are the "
        "data.  IRT places each student AND each question on the same "
        "ability scale -- a hard question contributes more information "
        "than an easy one, and a student gets an estimated theta with "
        "its own standard error.  Better than summing 0/1s when the "
        "questions vary in difficulty."
    ),
    "Copula": (
        "Use when the *marginal* distributions are different shapes "
        "but they move together.  A copula factors any joint into "
        "(margins) x (dependence structure), so you can model income "
        "as lognormal, age as truncated normal, and their tail "
        "dependence as a Clayton copula -- without forcing the joint "
        "into a Gaussian everything."
    ),
    "Network": (
        "Use when the *relationships between observations* are the "
        "thing.  Centrality answers 'who's important?', community "
        "detection answers 'who hangs out with whom?', and shortest "
        "path answers 'how do I get from A to B?'.  If your data fits "
        "in a CSV with no edge column, you probably don't need this."
    ),
    "Causal": (
        "Use when you want to answer 'what would happen if?', not "
        "just 'what's correlated?'.  ATE, IPW, DML, IV -- all of "
        "these try to clean confounding out of the answer.  Cheap "
        "rule of thumb: if the result would change a clinical "
        "guideline, you needed a causal estimator, not a regression."
    ),
    "Bayesian": (
        "Use when you have prior information worth keeping.  A "
        "frequentist 95% CI is 'over many runs, 95% catch the truth'; "
        "a Bayesian posterior is 'given what I knew + what I just "
        "saw, here is the distribution of plausible values'.  Pick "
        "Bayesian when the second framing answers your actual question."
    ),
    "Spatial": (
        "Use when 'closer is more similar' is plausible.  House "
        "prices, disease incidence, soil pH -- neighbouring locations "
        "share unobserved drivers (school district, water table) that "
        "violate the iid assumption.  Variograms quantify it; kriging "
        "predicts at unobserved locations; LM-lag/LM-err tests tell "
        "you which spatial spec your residuals call for."
    ),
    "TimeSeries": (
        "Use when 'now' is correlated with 'a moment ago'.  Yesterday's "
        "price predicts today's price; last week's flu cases predict "
        "this week's.  Unit-root + cointegration + GARCH are the "
        "starting trio -- find the trend, decide whether it's stochastic "
        "or deterministic, then model the volatility separately if it "
        "clusters."
    ),
    "CausalDID": (
        "Use when you want to estimate the *effect of being treated* "
        "by comparing how outcomes changed for treated vs untreated "
        "units across time.  The classic 2x2 design is fine when "
        "treatment timing is uniform; for staggered rollout (some "
        "units treated earlier than others) the post-2018 estimators "
        "(Goodman-Bacon, Callaway-Sant'Anna, Sun-Abraham, BJS) are "
        "what you want.  Always test parallel pre-trends."
    ),
    "CausalSyntheticControl": (
        "Use when you have one (or few) treated unit and many "
        "candidate controls -- and you want a data-driven weighted "
        "combination of controls that matches the treated unit's "
        "pre-period.  The post-treatment gap is your effect.  "
        "Better than a hand-picked control when there's no obvious "
        "single match (e.g., one country adopts a policy)."
    ),
    "CausalRDD": (
        "Use when treatment is determined by a sharp threshold on a "
        "running variable (test score, age, cutoff date).  Compare "
        "outcomes just-above and just-below the cutoff -- they should "
        "be exchangeable on everything except treatment.  Calonico-"
        "Cattaneo-Titiunik bandwidth + robust CIs are the modern "
        "default."
    ),
    "CausalML": (
        "Use when you want *heterogeneous* treatment effects (CATE) "
        "or when you have many controls and don't want to hand-pick.  "
        "Causal forests give you tau(x) for any covariate vector x; "
        "DML / TMLE / AIPW are doubly-robust mean-effect estimators "
        "that work even when one of the two nuisance models is wrong."
    ),
    "CausalSensitivity": (
        "Use AFTER you have a point estimate, to ask: how strong "
        "would unmeasured confounding need to be to overturn it?  "
        "E-value gives you the answer in plain RR units; Oster's "
        "delta uses observed-vs-unobserved bias proportionality; "
        "Cinelli-Hazlett gives you a benchmark against named "
        "covariates.  No serious causal claim ships without one of "
        "these."
    ),
    "Information": (
        "Use when 'how much do I know about X given Y?' is the "
        "question.  Entropy = uncertainty.  KL divergence = "
        "surprise (asymmetric distance between distributions).  "
        "Mutual information = how much knowing Y reduces "
        "uncertainty about X.  These are the building blocks for "
        "MDL, decision trees, variational inference, and the entire "
        "field of compression."
    ),
    "Differential": (
        "Use when you need to release an aggregate statistic about "
        "people *without* leaking any individual's data.  Add "
        "calibrated noise (Laplace for epsilon-DP, Gaussian for "
        "(epsilon, delta)-DP), and you get a mathematical guarantee "
        "that a single record's presence/absence is bounded-detectable.  "
        "Standard tools: Laplace mechanism, Gaussian mechanism, "
        "exponential mechanism for categorical, randomised response "
        "for binary, k-anonymity / l-diversity for tabular release."
    ),
    "Survey": (
        "Use when your data isn't iid -- it came from a complex "
        "sampling design (stratified, clustered, weighted, "
        "post-stratified).  Treating it as iid biases SEs by "
        "factors of 2-10x.  Horvitz-Thompson + design effect "
        "(DEFF) + jackknife/BRR/Taylor variance estimators are the "
        "canonical toolkit.  CPADS / ACS / NHANES -- anything from "
        "a national stats agency -- needs this."
    ),
    "_default": (
        "Look up the reference at the bottom of this cheatsheet.  "
        "If you've never met this concept before, the paper's intro "
        "+ a Wikipedia article is faster than this one-liner."
    ),
}


def _pick(items: Iterable[str], key: str) -> str:
    """Deterministic pick from an iterable based on a key.  Same `key`
    always picks the same item -- keeps cheatsheets stable across
    invocations rather than rolling the dice each time."""
    items = list(items)
    if not items:
        return ""
    h = sum(ord(c) for c in key)
    return items[h % len(items)]


# ── lazy spec-file index ────────────────────────────────────────────────────
# The generator that creates fn/<name>.py from spec JSONs doesn't emit a
# CATEGORY constant in the file (each fn module only carries the formula,
# the params, and the citation). So we recover the mapping lazily by
# reading every scripts/equation_specs_*.json on first call.
#
# Built once and cached at module level -- the spec files don't change
# between fn calls, and a fresh process re-reads on cold start.

_SPEC_INDEX: dict[str, str] | None = None


def _spec_index() -> dict[str, str]:
    global _SPEC_INDEX
    if _SPEC_INDEX is not None:
        return _SPEC_INDEX
    out: dict[str, str] = {}
    # Walk up from this file to find scripts/equation_specs_*.json.
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "scripts"
        if candidate.is_dir() and any(candidate.glob("equation_specs_*.json")):
            for f in candidate.glob("equation_specs_*.json"):
                try:
                    raw = json.loads(f.read_text())
                except Exception:
                    continue
                for entry in raw if isinstance(raw, list) else []:
                    name = entry.get("name")
                    cat = entry.get("category")
                    if name and cat:
                        out[name] = cat
            break
    _SPEC_INDEX = out
    return out


def _category_for(fn_short: str, module) -> str | None:
    """Resolve category from (in order): module CATEGORY constant,
    'Category:' line in module docstring, the spec-file index built
    from scripts/equation_specs_*.json."""
    cat = getattr(module, "CATEGORY", None)
    if isinstance(cat, str):
        return cat
    doc = inspect.getdoc(module) or ""
    m = re.search(r"^Category:\s*([A-Za-z0-9_]+)", doc, re.MULTILINE)
    if m:
        return m.group(1)
    return _spec_index().get(fn_short)


def _prefix_match(category: str | None, table: dict[str, list[str] | str]):
    """Find the longest prefix of `category` that has an entry in `table`.
    Returns the entry, or `table["_default"]` when nothing matches."""
    if category:
        for prefix in sorted(table.keys(), key=len, reverse=True):
            if prefix == "_default":
                continue
            if category.startswith(prefix):
                return table[prefix]
    return table["_default"]


def _docstring_first_line(fn_callable) -> str:
    doc = inspect.getdoc(fn_callable) or ""
    return doc.splitlines()[0] if doc else ""


def _docstring_reference(fn_callable) -> str:
    """Pull a 'References' or trailing citation out of the docstring."""
    doc = inspect.getdoc(fn_callable) or ""
    m = re.search(r"References?\s*\n[-=]+\s*\n(.+)", doc, re.DOTALL)
    if m:
        return m.group(1).strip().splitlines()[0]
    # Last-line citation pattern (the generator emits this).
    last = doc.strip().splitlines()
    if last:
        return last[-1].strip()
    return ""


def cheatsheet(fn_name: str) -> str:
    """Multi-section enriched help block for a single fn/ entry.

    Falls back gracefully when bits are missing -- every section is
    independently optional, so a fn with only a one-line cheatsheet
    still gets a usable card (just without the "When to use" guidance).

    Parameters
    ----------
    fn_name : str
        The short import alias (e.g. ``"huberw"``, ``"icc1"``).

    Returns
    -------
    str
        Multi-line block ready to print to a terminal.
    """
    try:
        module = importlib.import_module(f"morie.fn.{fn_name}")
    except ImportError as e:
        return f"{fn_name}: not found ({e})"

    # 1. Short description -- prefer the module-level cheatsheet() call
    #    if defined, else fall back to the first docstring line of the
    #    public function.
    short = ""
    if hasattr(module, "cheatsheet") and callable(module.cheatsheet):
        try:
            short = str(module.cheatsheet())
        except Exception:
            short = ""
    if not short:
        # Find the first public function in the module and read its
        # docstring.
        for name, obj in vars(module).items():
            if name.startswith("_") or not callable(obj):
                continue
            short = _docstring_first_line(obj)
            if short:
                break

    # 2. Category lookup
    category = _category_for(fn_name, module)

    # 3. When-to-use guidance + quote
    guidance = _prefix_match(category, WHEN_TO_USE)
    quote_pool = _prefix_match(category, CATEGORY_QUOTES)
    quote = _pick(quote_pool, fn_name)

    # 4. Reference (formula source)
    ref_text = ""
    for name, obj in vars(module).items():
        if name.startswith("_") or not callable(obj):
            continue
        ref_text = _docstring_reference(obj)
        if ref_text:
            break

    parts: list[str] = []
    parts.append(f"━━━ {fn_name} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if short:
        parts.append(short)
        parts.append("")
    if category:
        parts.append(f"Category   {category}")
    parts.append(f"Import     from morie.fn import {fn_name}")
    if ref_text:
        parts.append(f"Reference  {ref_text}")
    parts.append("")
    parts.append("When to use this")
    # Wrap to 70 cols for readability.
    parts.append(_wrap(guidance, 70))
    parts.append("")
    if quote:
        parts.append(quote)
    return "\n".join(parts)


def _wrap(text: str, width: int) -> str:
    """Manual wrap so we don't pull in `textwrap` for its 1 function."""
    out: list[str] = []
    line = ""
    for word in text.split():
        if len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = (line + " " + word) if line else word
    if line:
        out.append(line)
    return "\n".join(out)


__all__ = ["cheatsheet", "CATEGORY_QUOTES", "WHEN_TO_USE"]
