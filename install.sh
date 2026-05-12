#!/usr/bin/env bash
# install.sh — one-shot installer for morie on Linux / macOS / Windows-WSL.
#
# "No lockjaw with morie."  The point of this script is to free users
# from dependency purgatory: if Python or R is missing, it offers to
# install them via the host's package manager FIRST, then installs the
# morie Python and R packages on top.  No tetanus from rusty tooling.
#
# Usage:
#   ./install.sh                   # interactive: check + install both
#   ./install.sh --auto            # non-interactive: auto-install missing deps
#   ./install.sh --with-kernel     # also build the symbolic GPL kernel module
#   ./install.sh --python-only     # skip R
#   ./install.sh --r-only          # skip Python
#   ./install.sh --check-only      # only check what's installed; install nothing
#   ./install.sh --dry-run         # show what would happen, don't execute
#
# Exit codes: 0 success | 1 install error | 2 bad argument | 3 user-cancel
#
# License: this installer script is dual MIT/Apache-2.0 (matches Python side).
set -eu

WITH_KERNEL=0; PY=1; R=1; AUTO=0; CHECK_ONLY=0; DRY=0
for arg; do
  case "$arg" in
    --with-kernel) WITH_KERNEL=1 ;;
    --python-only) R=0 ;;
    --r-only) PY=0 ;;
    --auto) AUTO=1 ;;
    --check-only) CHECK_ONLY=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '/^# Usage/,/^$/p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

run() { if [ "$DRY" = "1" ]; then echo "[dry-run] $*"; else "$@"; fi }
prompt() {
  # prompt user; returns 0 (yes) or 1 (no).  In --auto always yes.
  if [ "$AUTO" = "1" ]; then return 0; fi
  read -r -p "$1 [Y/n] " ans
  case "$ans" in n|N|no|NO) return 1 ;; *) return 0 ;; esac
}

# --- Detect OS + package manager ----------------------------------
OS="$(uname -s)"
PKGMGR=""
case "$OS" in
  Linux)
    if   command -v apt-get >/dev/null 2>&1; then PKGMGR="apt-get"
    elif command -v dnf     >/dev/null 2>&1; then PKGMGR="dnf"
    elif command -v pacman  >/dev/null 2>&1; then PKGMGR="pacman"
    elif command -v zypper  >/dev/null 2>&1; then PKGMGR="zypper"
    elif command -v apk     >/dev/null 2>&1; then PKGMGR="apk"
    fi ;;
  Darwin)
    if command -v brew >/dev/null 2>&1; then PKGMGR="brew"
    else PKGMGR="" ; fi ;;
  MINGW*|MSYS*|CYGWIN*)
    if command -v winget >/dev/null 2>&1; then PKGMGR="winget"
    elif command -v choco >/dev/null 2>&1; then PKGMGR="choco"
    fi ;;
esac

echo "[install.sh] OS: $OS, package manager: ${PKGMGR:-none}"

# --- Helpers to install Python and R via the detected manager ------
install_python() {
  case "$PKGMGR" in
    apt-get) run sudo apt-get update && run sudo apt-get install -y python3 python3-pip python3-venv ;;
    dnf)     run sudo dnf install -y python3 python3-pip ;;
    pacman)  run sudo pacman -Sy --noconfirm python python-pip ;;
    zypper)  run sudo zypper install -y python3 python3-pip ;;
    apk)     run sudo apk add python3 py3-pip ;;
    brew)    run brew install python ;;
    winget)  run winget install -e --id Python.Python.3.12 ;;
    choco)   run choco install -y python ;;
    "")
      echo "[install.sh] no supported package manager detected on $OS."
      if [ "$OS" = "Darwin" ]; then
        echo "  On macOS, install Homebrew first:"
        echo "    /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo "  Then re-run this script.  Alternatively, download Python from https://www.python.org/downloads/"
      else
        echo "  Please install Python 3.10+ manually from https://www.python.org/downloads/ and re-run."
      fi
      return 1 ;;
  esac
}

install_r() {
  case "$PKGMGR" in
    apt-get) run sudo apt-get update && run sudo apt-get install -y r-base r-base-dev ;;
    dnf)     run sudo dnf install -y R R-devel ;;
    pacman)  run sudo pacman -Sy --noconfirm r ;;
    zypper)  run sudo zypper install -y R-base R-base-devel ;;
    apk)     run sudo apk add R R-dev ;;
    brew)    run brew install --cask r ;;
    winget)  run winget install -e --id RProject.R ;;
    choco)   run choco install -y r.project ;;
    "")
      echo "[install.sh] no supported package manager detected on $OS."
      echo "  Please install R 4.3+ manually from https://cran.r-project.org/ and re-run."
      return 1 ;;
  esac
}

# --- Check Python -------------------------------------------------
HAVE_PY=0
if command -v python3 >/dev/null 2>&1; then
  PYV=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
  echo "[install.sh] OK python3 $PYV present"
  HAVE_PY=1
else
  echo "[install.sh] -- python3 NOT found"
fi

# --- Check R -----------------------------------------------------
HAVE_R=0
if command -v Rscript >/dev/null 2>&1; then
  RV=$(Rscript -e 'cat(paste(R.version$major, R.version$minor, sep="."))' 2>/dev/null)
  echo "[install.sh] OK Rscript $RV present"
  HAVE_R=1
else
  echo "[install.sh] -- Rscript NOT found"
fi

if [ "$CHECK_ONLY" = "1" ]; then
  echo "[install.sh] --check-only: exiting after presence check"
  exit 0
fi

# --- Bootstrap missing deps ---------------------------------------
if [ "$PY" = "1" ] && [ "$HAVE_PY" = "0" ]; then
  if prompt "Install Python 3 via ${PKGMGR:-system}?"; then
    install_python || { echo "[install.sh] Python install failed"; exit 1; }
    HAVE_PY=1
  else
    echo "[install.sh] skipping Python install (user declined)"
    PY=0
  fi
fi

if [ "$R" = "1" ] && [ "$HAVE_R" = "0" ]; then
  if prompt "Install R via ${PKGMGR:-system}?"; then
    install_r || { echo "[install.sh] R install failed"; exit 1; }
    HAVE_R=1
  else
    echo "[install.sh] skipping R install (user declined)"
    R=0
  fi
fi

# --- Python morie -------------------------------------------------
if [ "$PY" = "1" ] && [ "$HAVE_PY" = "1" ]; then
  echo "[install.sh] installing Python morie via pip"
  run python3 -m pip install --upgrade pip
  run python3 -m pip install --upgrade morie
fi

# --- R morie ------------------------------------------------------
if [ "$R" = "1" ] && [ "$HAVE_R" = "1" ]; then
  echo "[install.sh] installing R morie via CRAN + r-universe fallback"
  run Rscript -e 'install.packages("morie", repos = c(hadesllm = "https://hadesllm.r-universe.dev", CRAN = "https://cloud.r-project.org"))'
fi

# --- Kernel module (opt-in, Linux only) ---------------------------
if [ "$WITH_KERNEL" = "1" ]; then
  if [ "$OS" != "Linux" ]; then
    echo "[install.sh] kernel module is Linux-only; skipping on $OS"
  elif ! [ -d /lib/modules/"$(uname -r)"/build ]; then
    echo "[install.sh] kernel headers for $(uname -r) not found; install linux-headers"
    case "$PKGMGR" in
      apt-get) echo "  sudo apt-get install linux-headers-\$(uname -r)" ;;
      dnf)     echo "  sudo dnf install kernel-devel-\$(uname -r)" ;;
      pacman)  echo "  sudo pacman -S linux-headers" ;;
    esac
    exit 1
  else
    echo "[install.sh] building kernel module (symbolic GPL declaration)"
    run make -C /lib/modules/"$(uname -r)"/build M="$(pwd)/kernel-module" modules
    run sudo insmod kernel-module/morie.ko
    echo "[install.sh] verify: cat /sys/kernel/morie/version"
  fi
fi

echo "[install.sh] done.  No lockjaw with morie."
