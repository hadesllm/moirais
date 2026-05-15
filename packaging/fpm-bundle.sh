#!/usr/bin/env bash
# Build standalone Linux packages (.deb + .rpm) for morie from the
# PyInstaller bundle.
#
# Unlike packaging/fpm-deb.sh (which depends on system python3 and
# pip-installs morie into a venv at postinst), these packages EMBED an
# isolated Python interpreter. They need no system Python, no pip, and
# no network access at install time.
#
# Usage: packaging/fpm-bundle.sh <version> <bundle-dir> [arch]
#   <bundle-dir>  the dist/morie directory produced by morie.spec
#   arch          amd64 (default) | arm64
#
# Output: dist/installer/morie_<version>_<arch>.deb
#         dist/installer/morie-<version>.<rpmarch>.rpm
set -euo pipefail

VERSION="${1:?usage: fpm-bundle.sh <version> <bundle-dir> [arch]}"
BUNDLE="${2:?usage: fpm-bundle.sh <version> <bundle-dir> [arch]}"
ARCH="${3:-amd64}"

OUT="dist/installer"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT"

# Stage: the bundle lands at /opt/morie; /usr/bin/morie symlinks into it.
mkdir -p "$WORK/opt" "$WORK/usr/bin"
cp -R "$BUNDLE" "$WORK/opt/morie"
ln -sf /opt/morie/morie "$WORK/usr/bin/morie"

# deb uses amd64/arm64; rpm uses x86_64/aarch64.
case "$ARCH" in
  amd64) RPMARCH=x86_64 ;;
  arm64) RPMARCH=aarch64 ;;
  *)     RPMARCH="$ARCH" ;;
esac

common=(
  -s dir
  --name morie
  --version "$VERSION"
  --maintainer "Vansh Singh Ruhela <hadesllm@proton.me>"
  --url "https://github.com/hadesllm/morie"
  --license "MIT OR Apache-2.0"
  --vendor "hadesllm"
  --description "Multi-domain Open Research and Inferential Estimation (morie) -- standalone bundle, no system Python required."
  -C "$WORK"
)

fpm "${common[@]}" -t deb --architecture "$ARCH" \
  --deb-no-default-config-files \
  --package "$OUT/morie_${VERSION}_${ARCH}.deb" \
  opt usr

fpm "${common[@]}" -t rpm --architecture "$RPMARCH" \
  --package "$OUT/morie-${VERSION}.${RPMARCH}.rpm" \
  opt usr

echo "Built:"
ls -lh "$OUT"/morie*"${VERSION}"*
