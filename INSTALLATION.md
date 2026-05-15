# Installing morie

> ⚠️ **Pre-alpha (v0.x).** morie is in pre-alpha. The first alpha milestone is **v1.0.0**; everything before that is point-releases of pre-alpha code. APIs may shift, datasets may move, and findings may be refined between minor versions.

## Easiest: the click-through installer

If you just want morie working with no setup — no Python, no pip, no
winget — download the installer for your system from the
**[latest release](https://github.com/hadesllm/morie/releases/latest)**
and run it:

| System | File | How |
|---|---|---|
| **Windows** | `morie-*-windows-x64-setup.exe` | Double-click, follow the wizard. Unsigned — at the SmartScreen prompt click **More info → Run anyway**. The x64 build also runs on Windows on ARM via emulation. |
| **macOS** | `morie-*-macos-arm64.pkg` | Double-click, follow the installer (Apple Silicon). Intel Macs: use `pip install morie` or the one-liner below. |
| **Linux** | `morie_*_amd64.deb` / `morie_*_arm64.deb`, or `morie-*.x86_64.rpm` / `morie-*.aarch64.rpm` | `sudo apt install ./morie_*.deb` or `sudo dnf install ./morie-*.rpm` |

On Linux, pick the file matching your CPU architecture
(`amd64`/`x86_64` for Intel/AMD, `arm64`/`aarch64` for ARM).
The installer wraps a self-contained Python bundle. When it finishes,
open a terminal — on Windows, the **"morie Console"** Start Menu
shortcut opens one ready to go — and run `morie --help`.

This installs the morie **command-line tool**. To use morie as a
Python library (`import morie`) in your own scripts, use one of the
manual channels below instead.

## Manual install channels

For more control, or to use morie as a Python/R library, morie also
ships across six manual channels. Pick the one that matches your
environment.

| | Channel | When to use | Needs |
|---|---|---|---|
| 1 | **Curl one-liner** | Linux / macOS / WSL, no Python or pip yet | `curl`, `bash` |
| 2 | **Windows** | Native Windows 10 / 11 | nothing — official `.exe` installers |
| 3 | **Homebrew tap** | macOS or Linuxbrew users | Homebrew |
| 4 | **PyPI (pip)** | You already manage your own venv | Python ≥3.10, pip |
| 5 | **Docker (GHCR)** | Zero-install or CI/CD | Docker |
| 6 | **R (CRAN + r-universe)** | You want the R package | R ≥4.3 |

## 1. Curl one-liner (Linux / macOS / WSL)

The simplest path. Bootstraps everything you need — `uv` for managed Python, a venv at `~/.venvs/morie`, the morie wheel.

```bash
curl -fsSL https://hadesllm.github.io/morie/install.sh | bash
```

With R alongside Python:

```bash
curl -fsSL https://hadesllm.github.io/morie/install.sh | bash -s -- --auto
```

**What you get after install:**

- A shim at `~/.local/bin/morie` that resolves into the managed venv.
- The venv itself at `~/.venvs/morie` with Python 3.12 + the SciPy stack + morie.

**If `~/.local/bin` isn't on your `PATH`** (the installer warns when this is the case):

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Smoke test:**

```bash
morie --help
morie list-modules | head -5
```

## 2. Windows

Windows ships without `curl`, `bash`, `python`, or `R`, so the Linux/macOS curl one-liner above will not run there. There are two ways in. The **official installers** (2A) need no prerequisites and work on every Windows — this is the recommended path. **winget** (2B) is faster but is *not present on every Windows install*, so do not assume it. Run all terminal commands in **PowerShell** or **Windows Terminal**; WSL users should follow option 1 instead.

### 2A. Official installers (recommended — works everywhere)

**Python:**

1. Go to **[python.org/downloads](https://www.python.org/downloads/)** and download the latest Windows installer (the 64-bit build, or the ARM64 build if you are on an ARM PC).
2. Run the installer. **On the very first screen, tick "Add python.exe to PATH"** before clicking "Install Now". This is the single most-missed step — without it, `python` and `pip` are not found in any terminal.
3. Open **PowerShell** and install morie:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install morie
   ```

**R** (optional — only if you want the R package):

1. Go to **[cran.r-project.org/bin/windows/base](https://cran.r-project.org/bin/windows/base/)**, download the installer, and run it (the defaults are fine).
2. Open **PowerShell** and install morie from r-universe — it ships pre-compiled Windows binaries, so no Rtools is needed:

   ```powershell
   Rscript -e "install.packages('morie', repos=c('https://hadesllm.r-universe.dev', 'https://cloud.r-project.org'))"
   ```

**Smoke test:**

```powershell
python -c "import morie; print(morie.__version__)"
Rscript -e "library(morie); cat(as.character(packageVersion('morie')), '\n')"
```

### 2B. winget (faster — only if your Windows has it)

`winget` is Microsoft's package manager. It is bundled with current Windows 11, **but it is missing from many installs** — older Windows 10 builds, freshly imaged machines that have not run Store updates, Windows LTSC, and Windows Server. **Check first:**

```powershell
winget --version
```

- If that prints a version (e.g. `v1.8.x`), you can install the prerequisites with winget:

  ```powershell
  winget install -e --id Python.Python.3.12
  winget install -e --id RProject.R
  ```

  Close and reopen the terminal afterwards so `python.exe` / `Rscript.exe` land on `PATH`, then install morie with the `pip` / `Rscript` commands from 2A.

- If `winget --version` errors ("not recognized") or prints nothing, winget is not installed. Either install **"App Installer"** from the Microsoft Store (that provides `winget`) and reopen the terminal, or just use **option 2A** — it does not need winget at all.

### Known Windows gotchas

#### Typing `python` opens the Microsoft Store

Windows has app-execution aliases that hijack `python` and `python3` to open the Store. Disable them under **Settings → Apps → Advanced app settings → App execution aliases**, untick **python.exe** and **python3.exe**, then reopen the terminal.

#### "Execution of scripts is disabled on this system"

PowerShell's default execution policy can block pip's shim scripts. As Administrator:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Answer `Y` at the prompt, then reopen the terminal.

#### Long-path errors during `pip install`

Some morie dependencies hit Windows's default 260-character path limit. Enable long-path support once, as Administrator:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Reboot, then re-run `pip install morie`.

## 3. Homebrew (macOS / Linuxbrew)

```bash
brew tap hadesllm/morie
brew install morie
```

The tap source is at [`hadesllm/homebrew-morie`](https://github.com/hadesllm/homebrew-morie). It pulls morie's source distribution from PyPI and bundles a self-contained `python@3.12` venv at `$(brew --prefix)/opt/morie/libexec`. No system Python required.

**Upgrade later:**

```bash
brew update
brew upgrade morie
```

## 4. PyPI (manual `pip`)

This path assumes you already have a working Python ≥3.10 with `pip` and a venv.

```bash
python3 -m venv ~/.venvs/morie
source ~/.venvs/morie/bin/activate
pip install --upgrade pip
pip install morie
```

Or with the optional extras:

```bash
pip install "morie[interactive]"   # Terminal IDE (textual)
pip install "morie[carbon]"        # CodeCarbon emissions tracking
pip install "morie[ml]"            # imbalanced-learn for SMOTE
```

### Known PyPI gotchas

#### Debian / Ubuntu / Raspberry Pi OS — PEP 668

Modern Debian-family systems forbid pip outside of a venv:

```
error: externally-managed-environment
× This environment is externally managed
```

This is **expected behaviour** of the distro, not a morie bug. Always install inside a venv (option 1 or option 2 handle this for you).

#### Raspberry Pi OS 13 — `python3` segfaults on SciPy imports

The Pi's `/usr/bin/python3` is Python 3.13.5, which segfaults on importing the SciPy stack (a Debian-packaging issue). Symptoms:

```
$ python3 -c "import morie"
Segmentation fault (core dumped)
```

**Fix:** use the curl one-liner — it installs `uv` and creates a venv with Python 3.12, which works.

## 5. Docker (GHCR)

```bash
# Latest stable
docker run --rm ghcr.io/hadesllm/morie:latest morie --help

# Pin to a version for reproducibility
docker run --rm ghcr.io/hadesllm/morie:0.7.2 morie --help
```

The image is published on every release with both `:latest` and `:<version>` tags. Multi-arch (linux/amd64). Includes morie + the full SciPy + R stack + R 4.5.

## 6. R (CRAN + r-universe)

```r
# Stable from CRAN (when listing is live)
install.packages("morie")

# Nightly binary builds (recommended while CRAN listing is rolling out)
install.packages(
  "morie",
  repos = c(
    hadesllm = "https://hadesllm.r-universe.dev",
    CRAN     = "https://cloud.r-project.org"
  )
)
```

## Verifying the install

Any of these should print `morie 0.7.2`:

```bash
# Python
python -c "import morie; print(morie.__version__)"

# CLI
morie list-modules | head -3
```

```r
# R
library(morie)
packageVersion("morie")
```

## First analysis (≤60 seconds, no code to write)

After install, this exact sequence works on a fresh machine. No data download. No coding. Just copy-paste:

```bash
# 1. Sanity check
morie list-modules | head -5

# 2. Run a real power-analysis module on the bundled synthetic CPADS frame
morie run-module power-design --output-dir ~/morie-first-run/

# 3. Inspect the output CSVs
ls ~/morie-first-run/
head -3 ~/morie-first-run/power_summary.csv
```

The synthetic CPADS frame is a 1,200-row toy dataset bundled inside the wheel; module output will emit a `UserWarning` so you know the analysis isn't of real Statistics Canada microdata. When you have the real CPADS PUMF, pass its path: `morie run-module power-design --cpads-csv /path/to/real.csv`.

### In Python — `morie.datasets`

For analyses you want to write yourself, the top-level `morie.datasets` module gives one-call DataFrame loaders for the major Canadian sociolegal feeds:

```python
import morie.datasets as md

# Toronto Police Service open-data
mc = md.tps_major_crime(year=2024)       # ArcGIS pulled into a DataFrame
sh = md.tps_shootings(year=2024)
ho = md.tps_homicide(year=2024)

# Canadian Postsecondary Alcohol/Drug Use Survey
cpads = md.cpads()                       # real PUMF if available, else synth

# SIU director's reports (text-mining)
text   = md.siu_report_text("https://www.siu.on.ca/.../22-OFD-001.pdf")
fields = md.siu_report_fields(text)
print(fields["conclusion"])

# Any CKAN portal (Canada, UK, EU, Ontario, ...)
hits = md.ckan_search("https://open.canada.ca/data", "alcohol", rows=5)
```

Every helper returns a plain `pandas.DataFrame` — no bespoke result type, no boilerplate.

## Using your own dataset (any column names)

morie modules name canonical concepts ("weight", "alcohol_past12m", "age_group"…) but your dataset probably uses different names. You don't have to rename your columns — morie's schema layer handles it:

```python
import pandas as pd
import morie.schema as ms
from morie.cpads import CPADS_REQUIRED_VARIABLES

your_df = pd.read_csv("your-data.csv")  # has columns like 'wt', 'binge30', 'sex'

# Let morie figure out the mapping
mapping, scores = ms.infer_mapping(your_df, canonical=CPADS_REQUIRED_VARIABLES)
print(mapping)        # {'wt': 'weight', 'binge30': 'heavy_drinking_30d', 'sex': 'gender', ...}
canon_df = ms.apply_mapping(your_df, mapping)
# now canon_df has the canonical column names morie expects
```

Or be explicit if you don't trust the fuzzy match:

```python
canon_df = ms.apply_mapping(your_df, {
    "wt": "weight",
    "binge30": "heavy_drinking_30d",
    "sex": "gender",
})
```

The same works for any morie-supported domain: pass the canonical list for your module (`CPADS_REQUIRED_VARIABLES`, OTIS, TPS, etc.). Synonyms are pre-registered for common variants (`wt`/`wgt`/`pweight` → `weight`, `binge_30d`/`hed`/`heavy_binge` → `heavy_drinking_30d`, …).

## Languages

morie's CLI is bilingual (EN/FR) by default and ships translations for **English, French, Spanish, German, Mandarin (Simplified), Portuguese (pt-BR), Japanese, Arabic, Hindi**. Set `MORIE_LOCALE`:

```bash
MORIE_LOCALE=fr morie cheatsheet
MORIE_LOCALE=es morie doctor
MORIE_LOCALE=zh morie tutorial
```

Methodology documentation (the JSS papers, in-depth method descriptions) is English-only for now — translating dense statistical prose is its own scoped project. Help us add it: every locale is one dict edit at [`src/morie/i18n.py`](src/morie/i18n.py).

## Asking morie for help

Stuck? morie ships an LLM-backed help agent. From anywhere on the CLI:

```bash
morie ask "I have a treatment-control study with 200 students per arm — which module should I run?"

morie ask "What does the column 'ebac_legal' mean?"

morie ask "I got a FileNotFoundError on cpads-2021-2022. How do I fix it?"
```

The agent reads morie's own source + documentation and answers in plain English. It also routes to `morie tutorial` / `morie cheatsheet` / `morie explain <file>` when a structured answer fits better. No API key required for the local-model path; cloud routing (Gemini, Anthropic Claude) is opt-in via env vars (see `morie doctor`).

## Pulling open data with `morie ingest`

v0.5.0 ships three open-data adapters:

```bash
# CKAN portals (open.canada.ca, data.gov.uk, data.europa.eu, ...)
morie ingest ckan --portal https://open.canada.ca/data \
                  --search "alcohol"

morie ingest ckan --portal https://open.canada.ca/data \
                  --package canadian-postsecondary-alcohol-and-drug-use-survey \
                  --out ./cpads/

# Toronto Police Service ArcGIS open-data layers
morie ingest tps --list
morie ingest tps --layer major-crime --year 2024 \
                 --out tps-major-2024.csv

# Special Investigations Unit director's-report mining
morie ingest siu --list                                 # index → CSV
morie ingest siu --report-id 22-OFD-001 --out report/   # text + structured fields
```

Each adapter is also importable as `morie.ingest.{ckan,tps,siu}` for use inside Python scripts.

## Updating to a newer release

| Channel | Update command |
|---|---|
| curl one-liner | Re-run the curl command (installs the latest) |
| Homebrew | `brew update && brew upgrade morie` |
| pip | `pip install --upgrade morie` |
| Docker | `docker pull ghcr.io/hadesllm/morie:latest` |
| R | Re-run the `install.packages(...)` above |

## Removing morie

```bash
# curl one-liner
rm -rf ~/.venvs/morie ~/.local/bin/morie

# Homebrew
brew uninstall morie
brew untap hadesllm/morie  # optional — removes the tap repo cache

# pip
pip uninstall morie

# R
remove.packages("morie")
```

## Getting help

- Documentation: <https://hadesllm.github.io/morie/>
- Source / issues: <https://github.com/hadesllm/morie>
- PyPI: <https://pypi.org/project/morie/>
- r-universe: <https://hadesllm.r-universe.dev/morie>
- Homebrew tap: <https://github.com/hadesllm/homebrew-morie>
