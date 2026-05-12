#!/usr/bin/env bash
# Build an RPM for morie via fpm.
#
# Usage: ./packaging/fpm-rpm.sh <VERSION> [ARCH]
#   VERSION: morie version, e.g. 0.4.0a0
#   ARCH:    rpm architecture (x86_64 | aarch64), default x86_64
#
# Output: dist/morie-${VERSION}.${ARCH}.rpm
#
# Mirrors fpm-deb.sh: wheel is pip-installed into /opt/morie/venv by the
# RPM %post scriptlet; /usr/local/bin/morie is a thin bash launcher.

set -euo pipefail

VERSION="${1:?usage: fpm-rpm.sh <VERSION> [ARCH]}"
ARCH="${2:-x86_64}"

mkdir -p dist staging/rpm

# --- Stub launcher --------------------------------------------------------
mkdir -p staging/rpm/usr/local/bin
cat > staging/rpm/usr/local/bin/morie <<'LAUNCHER'
#!/usr/bin/env bash
VENV="/opt/morie/venv"
if [ ! -x "${VENV}/bin/python" ]; then
  echo "morie: virtualenv at ${VENV} is missing or broken." >&2
  exit 1
fi
exec "${VENV}/bin/python" -m morie "$@"
LAUNCHER
chmod +x staging/rpm/usr/local/bin/morie

# --- post / preun scriptlets ---------------------------------------------
mkdir -p staging/rpm-scripts
cat > staging/rpm-scripts/postinst <<POSTINST
#!/usr/bin/env bash
set -e
PY="\$(command -v python3 || true)"
if [ -z "\${PY}" ]; then
  echo "morie post: python3 is required but not found." >&2
  exit 1
fi
"\${PY}" -m venv /opt/morie/venv
/opt/morie/venv/bin/pip install --upgrade pip
/opt/morie/venv/bin/pip install "morie==${VERSION}"
POSTINST
chmod +x staging/rpm-scripts/postinst

cat > staging/rpm-scripts/prerm <<'PRERM'
#!/usr/bin/env bash
set -e
rm -rf /opt/morie/venv
PRERM
chmod +x staging/rpm-scripts/prerm

# --- fpm build ------------------------------------------------------------
fpm -s dir -t rpm \
  --name morie \
  --version "${VERSION}" \
  --architecture "${ARCH}" \
  --maintainer "Vansh Singh Ruhela <hadesllm@proton.me>" \
  --url "https://github.com/hadesllm/morie" \
  --license "MIT OR Apache-2.0" \
  --description "Multi-domain Open Research and Inferential Estimation (morie)." \
  --depends "python3 >= 3.10" \
  --depends "python3-pip" \
  --rpm-summary "morie: Multi-domain Open Research and Inferential Estimation" \
  --rpm-dist  "el" \
  --after-install staging/rpm-scripts/postinst \
  --before-remove staging/rpm-scripts/prerm \
  --package "dist/morie-${VERSION}.${ARCH}.rpm" \
  -C staging/rpm \
  usr

echo "Built: dist/morie-${VERSION}.${ARCH}.rpm"
ls -lh "dist/morie-${VERSION}.${ARCH}.rpm"
