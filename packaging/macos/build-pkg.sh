#!/bin/bash
# Build the macOS .pkg click-through installer for morie.
#
# Usage:
#   packaging/macos/build-pkg.sh <version> <bundle-dir>
#
# <bundle-dir> is the dist/morie directory produced by
# packaging/pyinstaller/morie.spec.
#
# Signing + notarization run only when the matching env vars are set;
# without them an unsigned .pkg is produced (fine for local testing):
#
#   MORIE_SIGN_APP       "Developer ID Application: NAME (TEAMID)"
#   MORIE_SIGN_PKG       "Developer ID Installer: NAME (TEAMID)"
#   MORIE_NOTARY_PROFILE  notarytool keychain profile name
#
# Output: dist/installer/morie-<version>.pkg
set -euo pipefail

VERSION="${1:?usage: build-pkg.sh <version> <bundle-dir>}"
BUNDLE="${2:?usage: build-pkg.sh <version> <bundle-dir>}"

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mktemp -d)"
OUT="dist/installer"
mkdir -p "$OUT"
trap 'rm -rf "$WORK"' EXIT

# 1. Stage the bundle as the payload root (maps to /usr/local/morie).
ROOT="$WORK/root"
mkdir -p "$ROOT"
cp -R "$BUNDLE"/. "$ROOT/"
cp "$HERE/morie-console.command" "$ROOT/morie-console.command"
chmod +x "$ROOT/morie-console.command"

# 2. Code-sign the bundle (only if a Developer ID is configured).
if [[ -n "${MORIE_SIGN_APP:-}" ]]; then
  echo "Signing bundle: $MORIE_SIGN_APP"
  find "$ROOT" -type f \( -name "*.dylib" -o -name "*.so" \) -exec \
    codesign --force --timestamp --options runtime --sign "$MORIE_SIGN_APP" {} +
  codesign --force --timestamp --options runtime --sign "$MORIE_SIGN_APP" "$ROOT/morie"
fi

# 3. Component package.
pkgbuild \
  --root "$ROOT" \
  --identifier com.hadesllm.morie \
  --version "$VERSION" \
  --install-location /usr/local/morie \
  --scripts "$HERE/scripts" \
  "$WORK/morie-component.pkg"

# 4. Distribution (wizard) package.
product_args=(--distribution "$HERE/distribution.xml"
              --package-path "$WORK"
              --resources "$HERE")
if [[ -n "${MORIE_SIGN_PKG:-}" ]]; then
  product_args+=(--sign "$MORIE_SIGN_PKG")
fi
productbuild "${product_args[@]}" "$OUT/morie-$VERSION.pkg"

# 5. Notarize + staple (only if a notarytool profile is configured).
if [[ -n "${MORIE_NOTARY_PROFILE:-}" ]]; then
  echo "Notarizing..."
  xcrun notarytool submit "$OUT/morie-$VERSION.pkg" \
    --keychain-profile "$MORIE_NOTARY_PROFILE" --wait
  xcrun stapler staple "$OUT/morie-$VERSION.pkg"
fi

echo "Built: $OUT/morie-$VERSION.pkg"
