#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-morie:latest}"
CONTEXT="$(cd "$(dirname "$0")/.." && pwd)"
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

printf "\n=== MORIE Docker Test Suite ===\n"
printf "Image:   %s\n" "$IMAGE"
printf "Context: %s\n\n" "$CONTEXT"

printf "--- Phase 1: Build ---\n"
if DOCKER_BUILDKIT=1 docker build -t "$IMAGE" "$CONTEXT" --file "$CONTEXT/Dockerfile" 2>&1 | tail -5; then
    result pass "Docker build succeeded"
else
    result fail "Docker build failed"
    printf "\nBuild failed — cannot continue.\n"
    exit 1
fi

printf "\n--- Phase 2: Security Audit ---\n"

WHOAMI=$(docker run --rm "$IMAGE" whoami 2>/dev/null || echo "root")
if [ "$WHOAMI" = "morieapp" ]; then
    result pass "Non-root user: $WHOAMI"
else
    result fail "Running as root (got: $WHOAMI)"
fi

HC=$(docker inspect "$IMAGE" --format='{{.Config.Healthcheck}}' 2>/dev/null || echo "")
if echo "$HC" | grep -q "REGISTRY"; then
    result pass "HEALTHCHECK configured (registry validation)"
else
    result fail "HEALTHCHECK missing or misconfigured"
fi

DIGEST=$(docker inspect "$IMAGE" --format='{{index .RepoDigests 0}}' 2>/dev/null || echo "local-build")
if echo "$DIGEST" | grep -q "sha256:" || [ "$DIGEST" = "local-build" ]; then
    result pass "Image has digest or is local build"
else
    result fail "No digest found"
fi

UID_CHECK=$(docker run --rm "$IMAGE" id -u 2>/dev/null || echo "0")
if [ "$UID_CHECK" != "0" ]; then
    result pass "UID is non-zero: $UID_CHECK"
else
    result fail "UID is 0 (root)"
fi

printf "\n--- Phase 3: Functional Tests ---\n"

if docker run --rm "$IMAGE" python3 -c "import morie; print('morie imported')" 2>/dev/null | grep -q "morie imported"; then
    result pass "morie package importable"
else
    result fail "morie package import failed"
fi

REG_COUNT=$(docker run --rm "$IMAGE" python3 -c "from morie.fn._registry import REGISTRY; print(len(REGISTRY))" 2>/dev/null || echo "0")
if [ "$REG_COUNT" -gt 2100 ] 2>/dev/null; then
    result pass "Registry has $REG_COUNT entries (>2100)"
else
    result fail "Registry has $REG_COUNT entries (expected >2100)"
fi

if docker run --rm "$IMAGE" morie list-modules 2>/dev/null | grep -q "data-wrangling"; then
    result pass "morie list-modules works"
else
    result fail "morie list-modules failed"
fi

if docker run --rm "$IMAGE" python3 -c "from morie.fn import dnorm, ate, batman, spidm, optms, neom; print('imports ok')" 2>/dev/null | grep -q "imports ok"; then
    result pass "Pop-culture fn/ imports (SW/DC/Marvel/TF/Matrix)"
else
    result fail "Pop-culture fn/ imports failed"
fi

if docker run --rm "$IMAGE" python3 -c "
from morie.fn import dnorm
r = dnorm(0.0)
assert r.success, 'dnorm failed'
print('dnorm ok')
" 2>/dev/null | grep -q "dnorm ok"; then
    result pass "dnorm(0.0) returns success"
else
    result fail "dnorm(0.0) failed"
fi

if docker run --rm "$IMAGE" python3 -c "
from morie.fn._registry import REGISTRY
cats = set(e.category for e in REGISTRY.values())
assert len(cats) > 80, f'Only {len(cats)} categories'
print(f'{len(cats)} categories ok')
" 2>/dev/null | grep -q "categories ok"; then
    result pass "90+ categories in registry"
else
    result fail "Category count too low"
fi

printf "\n--- Phase 4: Image Size ---\n"

SIZE=$(docker image inspect "$IMAGE" --format='{{.Size}}' 2>/dev/null || echo "0")
SIZE_MB=$((SIZE / 1024 / 1024))
printf "  Image size: %sMB\n" "$SIZE_MB"
if [ "$SIZE_MB" -lt 2000 ]; then
    result pass "Image under 2GB ($SIZE_MB MB)"
else
    result fail "Image over 2GB ($SIZE_MB MB)"
fi

printf "\n--- Phase 5: Compose Validation ---\n"

if docker compose -f "$CONTEXT/docker-compose.yml" config --quiet 2>/dev/null; then
    result pass "docker-compose.yml valid"
else
    result fail "docker-compose.yml invalid"
fi

printf "\n=== Results: %d/%d passed" "$PASS" "$TOTAL"
if [ "$FAIL" -gt 0 ]; then
    printf " (%d FAILED)" "$FAIL"
fi
printf " ===\n\n"

exit "$FAIL"
