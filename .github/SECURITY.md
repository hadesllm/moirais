# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in MORIE, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email: **hadesllm@proton.me**

You should receive a response within 72 hours. We will work with you to understand the issue and coordinate a fix before any public disclosure.

## Scope

The following are in scope:
- Python package code (`morie/`)
- R package code (`r-package/morie/`)
- Docker images and configurations
- CI/CD pipeline configurations
- Cryptographic implementations (educational, not production)

The following are **out of scope**:
- Third-party dependencies (report upstream)
- The `dev/autoresearch*/` experimental training code
- The `dev/codecarbon-upstream/` fork

## Cryptographic Code Disclaimer

The cryptographic implementations in `morie/crypto/` (ML-KEM, ML-DSA, NTRU, lattice primitives, etc.) are **educational and reference implementations only**. They are NOT intended for production use and have NOT been audited for side-channel resistance or constant-time guarantees.
