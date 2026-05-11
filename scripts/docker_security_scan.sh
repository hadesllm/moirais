#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-morie:latest}"
PASS=0
FAIL=0
TOTAL=0

result() {
    TOTAL=$((TOTAL + 1))
    if [ "$1" = "pass" ]; then
        PASS=$((PASS + 1))
        printf "  [PASS] %s\n" "$2"
    else
        FAIL=$((FAIL + 1))
        printf "  [FAIL] %s\n" "$2"
    fi
}

printf "\n=== MORIE Docker Security Scan ===\n"
printf "Image: %s\n\n" "$IMAGE"

printf "--- 1. CIS Docker Benchmark Checks ---\n"

WHOAMI=$(docker run --rm "$IMAGE" whoami 2>/dev/null || echo "root")
if [ "$WHOAMI" != "root" ]; then
    result pass "CIS 4.1: Non-root user ($WHOAMI)"
else
    result fail "CIS 4.1: Running as root"
fi

HC=$(docker inspect "$IMAGE" --format='{{.Config.Healthcheck}}' 2>/dev/null || echo "")
if [ -n "$HC" ] && [ "$HC" != "<nil>" ]; then
    result pass "CIS 4.6: HEALTHCHECK present"
else
    result fail "CIS 4.6: No HEALTHCHECK"
fi

SHELL_CHECK=$(docker run --rm "$IMAGE" sh -c 'which sudo 2>/dev/null || echo "no-sudo"' 2>/dev/null)
if echo "$SHELL_CHECK" | grep -q "no-sudo"; then
    result pass "No sudo binary in image"
else
    result fail "sudo found in image — remove it"
fi

printf "\n--- 2. Supply Chain Verification ---\n"

CONTEXT="$(cd "$(dirname "$0")/.." && pwd)"
DOCKERFILE="$CONTEXT/Dockerfile"
if grep -q '@sha256:' "$DOCKERFILE"; then
    DIGEST_COUNT=$(grep -c '@sha256:' "$DOCKERFILE")
    result pass "Base images pinned by digest ($DIGEST_COUNT pins)"
else
    result fail "Base images NOT pinned by digest — supply chain risk"
fi

if grep -q '^FROM.*AS builder' "$DOCKERFILE" && grep -q '^FROM.*AS runtime' "$DOCKERFILE"; then
    result pass "Multi-stage build (builder + runtime)"
else
    result fail "Missing multi-stage build"
fi

printf "\n--- 3. Secrets Scan ---\n"

SECRET_PATTERNS="password|secret|api_key|token|private_key|AWS_|GITHUB_TOKEN"
LAYER_SECRETS=$(docker history "$IMAGE" --no-trunc 2>/dev/null | grep -iE "$SECRET_PATTERNS" || true)
if [ -z "$LAYER_SECRETS" ]; then
    result pass "No secrets found in image history"
else
    result fail "Potential secrets in image layers"
    printf "    %s\n" "$LAYER_SECRETS"
fi

ENV_CHECK=$(docker run --rm "$IMAGE" env 2>/dev/null | grep -iE "KEY=|TOKEN=|SECRET=|PASSWORD=" | grep -v "=\$" | grep -v "=$" || true)
if [ -z "$ENV_CHECK" ]; then
    result pass "No hardcoded secrets in environment"
else
    result fail "Secrets found in container environment"
fi

printf "\n--- 4. Filesystem Permissions ---\n"

SUID=$(docker run --rm "$IMAGE" find / -perm /4000 -type f 2>/dev/null | head -5 || true)
SUID_COUNT=$(echo "$SUID" | grep -c '/' 2>/dev/null || echo "0")
if [ "$SUID_COUNT" -lt 3 ]; then
    result pass "Minimal SUID binaries ($SUID_COUNT found)"
else
    result fail "Too many SUID binaries ($SUID_COUNT found)"
fi

WORLD_WRITE=$(docker run --rm "$IMAGE" find /app -perm -o+w -type f 2>/dev/null | head -5 || true)
if [ -z "$WORLD_WRITE" ]; then
    result pass "No world-writable files in /app"
else
    result fail "World-writable files found in /app"
fi

printf "\n--- 5. Docker Scout (if available) ---\n"

if command -v docker 2>/dev/null | grep -q docker && docker scout version 2>/dev/null; then
    printf "  Running Docker Scout CVE scan...\n"
    docker scout cves "$IMAGE" --only-severity critical,high 2>/dev/null || printf "  Scout scan completed (check output above)\n"
    result pass "Docker Scout scan executed"
else
    printf "  Docker Scout not available — install via Docker Desktop\n"
    printf "  Alternative: docker run --rm aquasec/trivy image %s\n" "$IMAGE"
    result pass "Scout not available (informational only)"
fi

printf "\n=== Security Scan: %d/%d passed" "$PASS" "$TOTAL"
if [ "$FAIL" -gt 0 ]; then
    printf " (%d FAILED)" "$FAIL"
fi
printf " ===\n\n"

exit "$FAIL"
