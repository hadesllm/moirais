# r-universe configuration

The MORIE R package is registered with r-universe via the
**hadesllm** organization. r-universe rebuilds binaries for
Windows / macOS / Linux every 24 hours from the GitHub source —
no GitHub Action is needed on this side.

## One-time setup (already done if you see hadesllm.r-universe.dev)

1. Sign in at https://r-universe.dev with your GitHub account
   (the one that owns or admins `hadesllm/morie`).
2. Visit https://github.com/r-universe-org/help#how-to-create-your-personal-r-universe
   and follow the instructions to create an org-level repo named
   `<owner>.r-universe.dev` with a `packages.json` file.
3. Add an entry pointing at the morie repo:

   ```json
   [
     {
       "package": "morie",
       "url": "https://github.com/hadesllm/morie",
       "subdir": "r-package/morie",
       "branch": "main"
     }
   ]
   ```

   The `subdir` field is critical — the R package source lives
   nested inside the repo, not at the top level.

## Installation for end users

Once the package builds successfully on r-universe (you can
check the dashboard at `https://hadesllm.r-universe.dev`):

```r
install.packages(
  "morie",
  repos = c(
    hadesllm = "https://hadesllm.r-universe.dev",
    CRAN     = "https://cloud.r-project.org"
  )
)
```

This pulls the latest binary build for the user's platform,
falling back to CRAN for any dependency r-universe doesn't
mirror.

## Relationship to CRAN

r-universe is **not** a CRAN replacement — it serves the same
package source via daily binary builds, with much faster
turnaround. The CRAN submission is governed by the
`r-cmd-check.yml` workflow on the same repo, which must pass on
all five matrix cells (Linux release/devel/oldrel + macOS +
Windows) before submitting.
