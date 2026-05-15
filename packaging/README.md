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

Both packages are the standalone [PyInstaller](https://pyinstaller.org)
bundle, packaged by [fpm](https://github.com/jordansissel/fpm):

- `packaging/fpm-bundle.sh <VERSION> <bundle-dir> [amd64|arm64]`
  &rarr; `dist/installer/morie_<VERSION>_<ARCH>.deb`
  + `dist/installer/morie-<VERSION>.<RPMARCH>.rpm`

Each package installs:

- `/opt/morie/` &mdash; the PyInstaller bundle, including an embedded,
  isolated Python interpreter.
- `/usr/bin/morie` &mdash; symlink into the bundle.

No system Python, `pip`, or network access is needed at install time. These
are byte-identical to the `.deb`/`.rpm` attached to each GitHub Release, so
`apt install morie` and a direct Release download give the same package.

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
python -m pip install . pyinstaller
pyinstaller packaging/pyinstaller/morie.spec --noconfirm
./packaging/fpm-bundle.sh 0.7.2 dist/morie amd64
```
