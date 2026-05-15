# morie system packages

Signed `.deb` and `.rpm` packages of `morie` are published to the
[hadesllm/morie-repo](https://hadesllm.github.io/morie-repo/) GitHub Pages
site by the `release-debrpm.yml` GitHub Actions workflow on every `v*` tag.

## Install morie on Debian/Ubuntu

```bash
sudo install -d -m 0755 /etc/apt/keyrings
curl -fsSL https://hadesllm.github.io/morie-repo/key.gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/hadesllm.gpg
echo "deb [signed-by=/etc/apt/keyrings/hadesllm.gpg] https://hadesllm.github.io/morie-repo/apt stable main" \
  | sudo tee /etc/apt/sources.list.d/morie.list
sudo apt update && sudo apt install morie
```

Supported architectures: `amd64`, `arm64`.

## Install morie on Fedora/RHEL

```bash
sudo dnf config-manager --add-repo https://hadesllm.github.io/morie-repo/rpm/morie.repo
sudo dnf install morie
```

Or on older RHEL/CentOS:

```bash
sudo yum-config-manager --add-repo https://hadesllm.github.io/morie-repo/rpm/morie.repo
sudo yum install morie
```

Supported architectures: `x86_64`, `aarch64`.

## How the packages are built

Both flavours are produced from the morie Python wheel by [fpm](https://github.com/jordansissel/fpm):

- `packaging/fpm-deb.sh <VERSION> [amd64|arm64]` &rarr; `dist/morie_<VERSION>_<ARCH>.deb`
- `packaging/fpm-rpm.sh <VERSION> [x86_64|aarch64]` &rarr; `dist/morie-<VERSION>.<ARCH>.rpm`

Each package installs:

- `/usr/local/bin/morie` &mdash; thin bash launcher.
- `/opt/morie/venv/` &mdash; private virtualenv populated by the post-install
  scriptlet via `pip install morie==<VERSION>` (depends on system `python3
  >= 3.10`).

This keeps system Python untouched and lets us ship a single binary-agnostic
package per architecture.

## Signing

Every artifact is signed with the project GPG key:

- `.deb` &rarr; `dpkg-sig --sign builder`
- `.rpm` &rarr; `rpmsign --addsign`
- apt `Release` &rarr; detached + clearsigned (`Release.gpg` + `InRelease`)
- dnf `repomd.xml` &rarr; detached signature at `repomd.xml.asc`

The ASCII-armored public key is published at
`https://hadesllm.github.io/morie-repo/key.gpg`.

## Verifying a downloaded package manually

```bash
# .deb
dpkg-sig --verify morie_0.7.2_amd64.deb

# .rpm
rpm --import https://hadesllm.github.io/morie-repo/key.gpg
rpm --checksig morie-0.7.2.x86_64.rpm
```

## Local dry-run (no signing)

```bash
gem install --no-document fpm
python -m build --wheel --outdir dist/
./packaging/fpm-deb.sh 0.7.2 amd64
./packaging/fpm-rpm.sh 0.7.2 x86_64
```
