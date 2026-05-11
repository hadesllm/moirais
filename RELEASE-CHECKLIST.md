# MORIE release checklist

Three distribution channels: **PyPI** (Python), **CRAN** (R), and
**r-universe** (R, daily-built binaries from GitHub source).

## Common pre-flight (all channels)

1. Bump the version number in two places (must match):
   - `dev/sphinx/project/pyproject.toml` → `[project] version`
   - `dev/sphinx/project/libexec/config/tools/r-package/morie/DESCRIPTION` → `Version:`
2. Update `CHANGELOG.md` (or release notes in the GitHub Release).
3. Run the local test suites and confirm green.
4. Commit, push to main, wait for CI to go green.

## PyPI release

- Trusted Publisher OIDC is configured: **no API token needed**.
- Workflow: `.github/workflows/pypi-publish.yml`.
- Trigger: push a tag `v*.*.*` (e.g., `v0.1.0`).
- Procedure:
  ```bash
  git tag v0.1.0 && git push origin v0.1.0
  ```
- The workflow builds sdist + wheel, runs `twine check`, then
  publishes via `pypa/gh-action-pypi-publish@release/v1` with
  OIDC. The `pypi` GitHub Environment provides the publish gate.

### One-time PyPI Trusted Publisher config (already done)

At https://pypi.org/manage/project/morie/settings/publishing/:

| Field             | Value                  |
|-------------------|------------------------|
| Owner             | hadesllm               |
| Repository name   | morie                |
| Workflow filename | pypi-publish.yml       |
| Environment       | pypi                   |

End users install with:

```bash
pip install morie
```

## CRAN release

- Workflow: `.github/workflows/r-cmd-check.yml` (multi-OS check).
- The workflow must pass cleanly on all 5 matrix cells before
  submitting.
- Submission is **manual** (CRAN policy):
  1. From the R package directory, run
     `R CMD build .` to produce `morie_<version>.tar.gz`.
  2. Upload the tarball at https://cran.r-project.org/submit.html.
  3. Respond to the maintainer-confirmation email within 24 h.
- Common CRAN gotchas:
  - DESCRIPTION Title ≤ 65 chars, in title case, no period.
  - DESCRIPTION Description must end in a period; 4-8 sentences;
    no leading "MORIE provides..." (start with content).
  - All URLs in DESCRIPTION must resolve (200 OK).
  - No `library()` calls in package code; use `requireNamespace()`.
  - All `tests/` must complete in < 10 min on CRAN servers.

End users install with:

```r
install.packages("morie")
```

## r-universe release

- **No release action needed** — r-universe rebuilds nightly.
- Configuration is documented at `.github/r-universe-config.md`.
- Dashboard: https://hadesllm.r-universe.dev
- End users install with:

```r
install.packages(
  "morie",
  repos = c(
    hadesllm = "https://hadesllm.r-universe.dev",
    CRAN     = "https://cloud.r-project.org"
  )
)
```

## After every release

1. Tag the GitHub release with the same `v*.*.*` tag.
2. Confirm the Zenodo software DOI minted via the GitHub-Zenodo
   integration (see `.zenodo.json`).
3. Update the MRM paper's `[Ruhela2026RF]` citation key on Zenodo
   to reference the new release DOI.
4. Update `papers/ruhela-formulations-paper/zenodo-paper-metadata.yaml`
   `related_identifiers` with the new software DOI.
