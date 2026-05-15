# Tutorial — your first morie analysis (no Python knowledge required)

This tutorial walks you through one complete analysis from install to interpretable output. Every step is a copy-paste command. We explain what each command does and what each output file contains.

If you get stuck, the [troubleshooting section](#when-something-goes-wrong) at the bottom has the fix.

---

## What we're going to do

We'll answer the question: **"Among Canadian university students, how many would we need to survey to detect a 5% difference in heavy-drinking rates between men and women, with 80% power?"**

This is a real *power-analysis* question — the kind a graduate student or policy analyst would run before designing a survey. You don't need to know the statistics; morie does the math.

By the end you'll have ~13 CSV files, and we'll show you which one answers the question.

## Step 1 — install morie

If you haven't installed yet:

```bash
curl -fsSL https://hadesllm.github.io/morie/install.sh | bash
```

That's it. The installer puts the `morie` command on your `PATH`. If the installer warns about `PATH`, run the line it shows you and then start a new terminal window.

Check it worked:

```bash
morie --help
```

You should see a list of subcommands. If you see "command not found", see [troubleshooting](#when-something-goes-wrong).

## Step 2 — what can morie do?

```bash
morie list-modules
```

This prints every analysis module morie ships. Each module:
- Has a short description of what it computes
- Lists the output files it writes

Look for `power-design` in the list. That's the module we're running.

## Step 3 — run the analysis

```bash
mkdir -p ~/morie-tutorial
morie run-module power-design --output-dir ~/morie-tutorial
```

What just happened:
1. `mkdir -p ~/morie-tutorial` made a folder in your home directory for the output.
2. `morie run-module power-design ...` ran the power-analysis module on the bundled toy CPADS dataset.

You'll see a `UserWarning` that the data is synthetic. This is **expected** — your first run uses a 1,200-row toy dataset so you can see how the tool works without downloading anything. We'll show you how to swap in real data later.

When it finishes (a few seconds), check what landed:

```bash
ls ~/morie-tutorial/
```

You should see ~13 CSV files.

## Step 4 — read the output

The CSV that answers our original question is `power_two_proportion_gender.csv`. Open it:

```bash
head -20 ~/morie-tutorial/power_two_proportion_gender.csv
```

The first column is the **effect size you want to detect** (the difference in heavy-drinking rates between men and women). The last column is the **sample size per group** you'd need. So if you want to detect a 5% gap, look at the row where `effect_size = 0.05` and read the `n_per_group` value.

Other useful CSVs in the same folder:

| File | What it tells you |
|---|---|
| `power_summary.csv` | One-row summary of the design parameters and recommendations |
| `power_two_proportion_gender.csv` | Sample-size grid for two-proportion tests by gender |
| `power_one_proportion_grid.csv` | Sample-size grid for a single-proportion question |
| `power_ebac_endpoint_anchors.csv` | Sample sizes for blood-alcohol-content endpoints |
| `power_interaction_*` | Sample sizes when looking at interaction effects (e.g. gender × age) |
| `randomization_*` | Pre-baked randomization blueprints for running the survey |

You don't need to write any code to use these — open them in Excel, Numbers, Google Sheets, R, or whatever you already use.

## Step 5 — swap in real data when you're ready

The synthetic CPADS frame is fine for learning the workflow, but you'll want real data for a real analysis. Here's the path:

1. Get a Statistics Canada account at <https://open.canada.ca/data> (free for academic use). You're looking for the **Public Use Microdata File** — abbreviated *PUMF* — which is the anonymised individual-record version of the survey.
2. Download the CPADS 2021-2022 PUMF CSV.
3. Re-run morie with the `--cpads-csv` flag:

```bash
morie run-module power-design \
    --cpads-csv ~/Downloads/cpads-2021-2022-pumf2.csv \
    --output-dir ~/morie-tutorial-real
```

That's the only change.

## Step 6 — what else can you do?

morie ships **23 modules**. Each one answers a specific kind of question. A few examples:

| Module | Question it answers |
|---|---|
| `power-design` | How many participants do I need? |
| `descriptive-statistics` | What does my data look like? |
| `frequentist-inference` | Is the effect I'm seeing real, or could it be chance? |
| `bayesian-inference` | What's the probability the effect is in a given range? |
| `logistic-models` | What predicts whether someone heavy-drinks? |
| `regression-models` | How much does X change Y? |

Run any of them the same way:

```bash
morie run-module descriptive-statistics --output-dir ~/morie-tutorial
morie run-module frequentist-inference --output-dir ~/morie-tutorial
```

The `morie doctor` command also runs all health checks at once:

```bash
morie doctor
```

This is useful when something isn't working — it tells you which dependency is missing.

## Step 7 — Toronto-Police-Service crime data

morie can pull TPS open data directly. From the command line:

```bash
# List the open-data layers TPS publishes
python -c "import morie.datasets as md; print(md.tps_layers().to_string(index=False))"

# Pull last year's major crime to a CSV
python -c "
import morie.datasets as md
md.tps_major_crime(year=2024).to_csv('tps-2024.csv', index=False)
"
```

If `python -c "..."` looks intimidating, the same is doable from a Python REPL (`morie repl`) or from a notebook — the point is that **you don't write more than the one line that names what you want**.

## When something goes wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| `morie: command not found` | `~/.local/bin` isn't on your `PATH` | `export PATH="$HOME/.local/bin:$PATH"` and add that line to your `~/.bashrc` or `~/.zshrc` |
| `error: externally-managed-environment` | Newer Debian / Ubuntu / Raspberry Pi systems don't let you install Python packages directly any more — they want each project in its own isolated folder (called a *virtual environment* or *venv*). This is a safety rule the OS added, not a morie bug. | Use the curl one-liner (Step 1); it creates the isolated folder for you. |
| `Segmentation fault` after `import morie` | The system Python on Raspberry Pi OS 13 has a known bug that crashes when loading scientific libraries. | Same fix: curl one-liner — it installs a working Python 3.12 alongside the broken system one. |
| `FileNotFoundError ... cpads-2021-2022-pumf2.csv` | (Pre-v0.5.0 only) — should not happen on v0.5.0+ which falls back to synthetic data | `python -c "import morie; print(morie.__version__)"` should show `0.7.2` or later; upgrade via `pip install -U morie` or re-run the curl installer |
| `UserWarning: using the SHIPPED SYNTHETIC CPADS frame` | Expected on first run; outputs are toy | Get real PUMF (Step 5) when ready |
| Some other error | We want to hear about it | File an issue at <https://github.com/hadesllm/morie/issues> — paste the full error message |

## What to do next

- Read the [INSTALLATION.md](INSTALLATION.md) for the full install matrix
- Browse [the docs site](https://hadesllm.github.io/morie/) for module reference
- Look at the [JSS papers](papers/) for the methodological backing
- Open the Terminal IDE for an interactive exploration: `morie tui`

If this is your first time writing code: **welcome.** Statistical computing is just a vocabulary you don't yet have. Each one of those 13 CSVs answers a specific question; you don't need to understand the implementation to read the answer. Keep going — every senior researcher you've read once typed `morie list-modules` for the first time.
