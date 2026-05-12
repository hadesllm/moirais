#!/usr/bin/env bash
# Build a Debian .deb for morie via fpm.
#
# Usage: ./packaging/fpm-deb.sh <VERSION> [ARCH]
#   VERSION: morie version, e.g. 0.4.0a0
#   ARCH:    deb architecture (amd64 | arm64), default amd64
#
# Output: dist/morie_${VERSION}_${ARCH}.deb
#
# The package wraps the morie wheel installed via pip into /opt/morie/venv
# and drops a thin launcher at /usr/local/bin/morie that execs the venv's
# `python -m morie`.  This keeps system Python untouched.

set -euo pipefail

VERSION="${1:?usage: fpm-deb.sh <VERSION> [ARCH]}"
ARCH="${2:-amd64}"

mkdir -p dist staging/deb

# --- Stub launcher --------------------------------------------------------
mkdir -p staging/deb/usr/local/bin
cat > staging/deb/usr/local/bin/morie <<'LAUNCHER'
#!/usr/bin/env bash
# morie launcher: defers to the bundled virtualenv.
VENV="/opt/morie/venv"
if [ ! -x "${VENV}/bin/python" ]; then
  echo "morie: virtualenv at ${VENV} is missing or broken." >&2
  exit 1
fi
exec "${VENV}/bin/python" -m morie "$@"
LAUNCHER
chmod +x staging/deb/usr/local/bin/morie

# --- postinst: create the venv and pip-install morie ----------------------
mkdir -p staging/deb-scripts
cat > staging/deb-scripts/postinst <<POSTINST
#!/usr/bin/env bash
set -e
PY="\$(command -v python3 || true)"
if [ -z "\${PY}" ]; then
  echo "morie postinst: python3 is required but not found." >&2
  exit 1
fi
"\${PY}" -m venv /opt/morie/venv
/opt/morie/venv/bin/pip install --upgrade pip
/opt/morie/venv/bin/pip install "morie==${VERSION}"
POSTINST
chmod +x staging/deb-scripts/postinst

cat > staging/deb-scripts/prerm <<'PRERM'
#!/usr/bin/env bash
set -e
rm -rf /opt/morie/venv
PRERM
chmod +x staging/deb-scripts/prerm

# --- fpm build ------------------------------------------------------------
fpm -s dir -t deb \
  --name morie \
  --version "${VERSION}" \
  --architecture "${ARCH}" \
  --maintainer "Vansh Singh Ruhela <hadesllm@proton.me>" \
  --url "https://github.com/hadesllm/morie" \
  --license "MIT OR Apache-2.0" \
  --description "Multi-domain Open Research and Inferential Estimation (morie)." \
  --depends "python3 (>= 3.10)" \
  --depends "python3-venv" \
  --depends "python3-pip" \
  --after-install staging/deb-scripts/postinst \
  --before-remove staging/deb-scripts/prerm \
  --deb-no-default-config-files \
  --package "dist/morie_${VERSION}_${ARCH}.deb" \
  -C staging/deb \
  usr

echo "Built: dist/morie_${VERSION}_${ARCH}.deb"
ls -lh "dist/morie_${VERSION}_${ARCH}.deb"
