# hadesllm — Security disclosure policy

This document is intended for the `hadesllm/.github` repository at
`SECURITY.md`.

## Reporting a vulnerability

Email <hadesllm@proton.me> with subject prefixed `[SECURITY]`.
ProtonMail provides end-to-end encryption to ProtonMail recipients.

For non-ProtonMail senders: encrypt against the project PGP key
published at `morie/SECURITY.md` and on the keyserver (when the
keyserver is public).

## Disclosure timeline

- Initial acknowledgement within **48 hours** (best-effort; sole
  maintainer until v0.4.0-alpha).
- Triage + severity assessment within **7 days**.
- Public disclosure coordinated with reporter; default **90 days**
  after first acknowledgement.

## Scope

In-scope:
- Code-execution paths in `morie` Python or R packages.
- Network paths in `keyserver`, `siu_fetch`, `tps_fetch`, or any
  scraper that retrieves external data.
- The Linux kernel module (`kernel-module/morie.c`) — though it
  intentionally provides no privileged surface.

Out-of-scope:
- Vulnerabilities in *upstream* dependencies (please report to those
  projects directly; we may track impact).
- Issues with the dev-time `polyglot.py` multi-language harness when
  used to execute arbitrary code (that surface is opt-in).

## Public PGP key

```
-----BEGIN PGP PUBLIC KEY BLOCK-----
(Pending — fetch from keyserver.hadesllm.com when keyserver goes
public.  Currently pre-alpha; email is the primary channel.)
-----END PGP PUBLIC KEY BLOCK-----
```
