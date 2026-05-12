# Contributing to MORIE

Thanks for considering a contribution. MORIE — *Multi-domain Open
Research and Inferential Estimation* — is an open-source toolkit for
observational inference in Python and R, and contributions of all
sizes are welcome.

## Quick orientation

- **Code of conduct.** Be kind. Disagree about the work, not the
  person. There is no formal CoC at this stage; the standard expected
  is "treat collaborators the way you'd want a methodological reviewer
  to treat you."
- **Maintainer.** Vansh Singh Ruhela (`@rootcoder007`,
  <hadesllm@proton.me>).
- **Licence.** MORIE uses per-component licensing (Python: `MIT OR Apache-2.0`; R + kernel module: `GPL-2.0-only`).  See `LICENSING.md`,
  `LICENSING.md`, and `LICENSING_ANALYSIS.md`.

## Where to start

### Looking for a first issue?

The `good-first-issue` label on the issue tracker is curated for
contributors who want to land a single, well-scoped change. Pick one,
comment to claim it, and ask any clarifying questions in the issue
thread before writing code.

If nothing is labelled when you look, that means the maintainer hasn't
caught up — you can:

- File an issue with a proposed change and ask "would a PR for this
  be welcome?", or
- Pick anything from the `help-wanted` label, or
- Pick anything else and check in on the design before writing it.

### What kinds of contributions we want

- **Bug fixes** — including failing-test reproductions for issues
  filed by others.
- **New statistical methods** that are: (a) well-cited, (b) have a
  unique signature distinct from existing functions, (c) come with
  unit tests and a docstring example. Drop a note in an issue first
  so we can pick the right module and the right Python/R parity.
- **R parity for Python-only functions** (or vice versa). The
  ledger of what exists where is in
  `r-package/morie/tests/testthat/test-parity-wrappers.R` and
  `tests/test_causal_parity.py`.
- **Documentation** — fixed typos, clearer examples, fleshed-out
  vignettes.
- **Translation / i18n** — for the TUI strings if you read a
  language we don't yet support.

### What kinds we don't want (right now)

- Wholesale rewrites of existing modules. Open an issue first.
- Vendoring new third-party source into the tree. We `Imports:` /
  `import` rather than vendor; see `LICENSING_ANALYSIS.md`.
- New language ports. Python and R are the supported front-ends.
- Personal scripts that aren't reusable.

## Development setup

### Python

```bash
git clone https://github.com/hadesllm/morie
cd morie
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,docs]"
python -m pytest tests/ -q
```

### R

```bash
cd r-package/morie
R CMD INSTALL --preclean .
Rscript -e 'testthat::test_local()'
```

### Documentation (Sphinx)

```bash
pip install -e ".[docs]"
sphinx-build -b html docs/source docs/build/html
# open docs/build/html/index.html
```

## Pull-request workflow

1. **Fork** the repo on GitHub.
2. **Branch** off `main`. Use a short topical name:
   `fix-irm-error-message`, `add-aipw-r-wrapper`,
   `docs-fix-typo-in-mrm-page`.
3. **Write the change.** Keep PRs focused — one logical change per
   PR. Multi-purpose PRs are hard to review.
4. **Add or update tests.** Every code change needs at least one
   test that would fail without the change. Pure-doc PRs are
   exempt.
5. **Run the test suite locally** (`python -m pytest`,
   `R CMD check`) and the linters (`ruff check .` for Python,
   `lintr::lint_package()` for R).
6. **Open the PR** against `hadesllm/morie:main`. Fill the PR
   template. Reference the issue you're closing
   (`Closes #123`).
7. **CI** runs the Python and R checks. If something is red, fix
   it locally and push again — don't ignore the result.
8. **Review.** The maintainer or another committer will leave
   comments. Discussion happens in the PR thread. Don't take
   review feedback personally — methodology software is held to
   a higher bar than most general-purpose libraries.
9. **Merge.** The maintainer squashes-and-merges. You don't need
   to rebase manually unless asked.

## Coding conventions

### Python

- 4-space indent, `ruff`-formatted, line length 120.
- Type hints on all public functions.
- Docstrings are NumPy-style with `Parameters`, `Returns`,
  `Examples` sections. Public functions need at least one
  runnable docstring example.
- Avoid abbreviations in identifier names. The exception is
  `morie.fn/` where short stat-textbook names (`dnorm`, `dfa`,
  `hfd`) are intentional.
- Tests live in `tests/`. Use `pytest`. Tests must be
  deterministic — seed any RNG explicitly.

### R

- 2-space indent, `lintr`-clean, line length 100.
- Roxygen2 documentation. Every exported function needs an
  `@examples` block.
- Tests live in `r-package/morie/tests/testthat/`. Use
  `testthat`.
- Imports are declared in `DESCRIPTION` under `Imports:` (hard
  dep) or `Suggests:` (soft / optional dep). No
  `library()` calls inside package code; use `pkg::fn()`.

### Documentation prose

- Sentence-case headings.
- Cross-reference functions with the standard role syntax:
  `:py:func:`morie.causal.estimate_att`` (Python),
  `:r:func:`morie::estimate_att`` (R).
- Maths in `.. math::` blocks (Sphinx) or `$...$` (Markdown).

## Reporting bugs

File at <https://github.com/hadesllm/morie/issues/new?template=bug.yml>.
Include:

- the morie version (`morie.__version__` or `packageVersion("morie")`),
- the language (Python/R),
- a minimal reproducer (the smallest code that fails),
- the actual vs expected behaviour,
- the platform (`uname -a`, `R.version.string` or
  `python --version`).

## Requesting a feature

File at <https://github.com/hadesllm/morie/issues/new?template=feature.yml>.
Sketch the use case and, if possible, the proposed API. The
maintainer responds within a few days.

## Security

Security disclosures: see `.github/SECURITY.md`. Do not file
security issues as public GitHub issues.

## Acknowledging contributors

Every merged PR gets a co-author trailer in the merge commit
when the contributor uses a noreply GitHub email. We also list
substantial contributors in `ACKNOWLEDGMENTS.md`.

Thanks for being here.

## Branch protection (maintainer setup)

The `main` branch is configured (as of v0.3.0) with a **permissive
contributor-friendly ruleset**:

- All PRs must pass the GitHub Actions `CI` workflow matrix
  (Python × {Ubuntu, macOS, Windows} × Py 3.10 + 3.14;
  R × {Ubuntu, macOS, Windows} × R release + devel + oldrel-1).
- All PRs must pass the GitHub Actions `R CMD check` workflow.
- One maintainer review is required on PRs from external contributors.
- Owners (currently @rootcoder007) can push directly to `main` for
  hotfix scenarios.
- Force-push to `main` is disabled.

The intent is: **if the bot says the build works, contribution
should work**.  CI is the gate, not human gatekeeping.  Owner-direct-
push exists only for hotfix scenarios; the normal flow for everything
including the maintainer is open PR → CI green → review → merge.

To configure these rules on a fork:
1. Settings → Branches → Add rule for `main`.
2. Require status checks: `CI`, `R CMD check`, `Build and Publish Container`.
3. Require pull request before merging: 1 approving review (from
   maintainers only).
4. Do not check "Restrict who can push to matching branches" so that
   forks contributors are not pre-emptively blocked.
