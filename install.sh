#!/usr/bin/env bash
# install.sh — one-shot installer for morie on Linux/macOS.
#
# Installs the Python and R surfaces of `morie` and optionally builds
# the symbolic GPL kernel module.  This is the v0.3.0 installer.
# A native DEB/RPM packaging pipeline is queued for v0.4.0.
#
# Usage:
#   ./install.sh                  # Python + R only
#   ./install.sh --with-kernel    # also build + insmod the kernel module
#   ./install.sh --python-only    # skip the R bits
#   ./install.sh --r-only         # skip the Python bits
#   ./install.sh --dry-run        # show what would happen
#
# License: see LICENSING.md.  The installer script itself is dual MIT/Apache.
set -eu

WITH_KERNEL=0; PY=1; R=1; DRY=0
for arg; do
  case "$arg" in
    --with-kernel) WITH_KERNEL=1 ;;
    --python-only) R=0 ;;
    --r-only) PY=0 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '/^# Usage/,/^$/p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

run() { if [ "$DRY" = "1" ]; then echo "[dry-run] $*"; else "$@"; fi }

# --- Python side -------------------------------------------------
if [ "$PY" = "1" ]; then
  if command -v python3 >/dev/null 2>&1; then
    echo "[install.sh] Python morie via pip"
    run python3 -m pip install --upgrade pip
    run python3 -m pip install morie
  else
    echo "[install.sh] WARNING: python3 not found; skipping Python install"
  fi
fi

# --- R side -------------------------------------------------------
if [ "$R" = "1" ]; then
  if command -v Rscript >/dev/null 2>&1; then
    echo "[install.sh] R morie via CRAN + r-universe fallback"
    run Rscript -e 'install.packages("morie", repos = c(hadesllm = "https://hadesllm.r-universe.dev", CRAN = "https://cloud.r-project.org"))'
  else
    echo "[install.sh] WARNING: Rscript not found; skipping R install"
  fi
fi

# --- Kernel module (opt-in) ---------------------------------------
if [ "$WITH_KERNEL" = "1" ]; then
  if [ "$(uname)" != "Linux" ]; then
    echo "[install.sh] kernel module is Linux-only; skipping on $(uname)"
  elif ! [ -d /lib/modules/"$(uname -r)"/build ]; then
    echo "[install.sh] kernel headers for $(uname -r) not found; install linux-headers"
    echo "  Debian/Ubuntu: sudo apt install linux-headers-\$(uname -r)"
    echo "  Fedora/RHEL : sudo dnf install kernel-devel-\$(uname -r)"
    exit 1
  else
    echo "[install.sh] building kernel module (symbolic GPL declaration)"
    run make -C /lib/modules/"$(uname -r)"/build M="$(pwd)/kernel-module" modules
    run sudo insmod kernel-module/morie.ko
    echo "[install.sh] verify: cat /sys/kernel/morie/version"
  fi
fi

echo "[install.sh] done."
