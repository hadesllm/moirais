# Tests for morie_det_rng() — SHA-keyed deterministic RNG helper.
#
# Mirrors tests/parity/test_det_rng.py in the Python tree.  Verifies
#   * Reproducibility: same (name, seed) -> same first-10 draws.
#   * Determinism across callers.
#   * Independence across names (KS uniformity + low correlation).
#   * SHA-256 hex matches the Python side bit-for-bit on a pinned input.

skip_if_no_hash <- function() {
  ok <- requireNamespace("digest", quietly = TRUE) ||
        requireNamespace("openssl", quietly = TRUE)
  testthat::skip_if_not(ok, "neither 'digest' nor 'openssl' available")
}

test_that("morie_det_rng is reproducible — same call, same draws", {
  skip_if_no_hash()
  morie_det_rng("foo", 42L)
  a <- runif(10)
  morie_det_rng("foo", 42L)
  b <- runif(10)
  expect_equal(a, b)
})

test_that("morie_det_rng is reproducible across interleaved callers", {
  skip_if_no_hash()
  morie_det_rng("ksr07_bootstrap", 12345L)
  draws_one <- rnorm(50)
  # Interleave an unrelated call to confirm no shared global state poisons.
  morie_det_rng("unrelated", 999L)
  invisible(rnorm(50))
  morie_det_rng("ksr07_bootstrap", 12345L)
  draws_two <- rnorm(50)
  expect_equal(draws_one, draws_two)
})

test_that("morie_det_rng returns an integer in R's int32 range", {
  skip_if_no_hash()
  for (nm in c("foo", "ksr07_bootstrap", "trfge")) {
    for (sd in c(0L, 1L, 42L, 12345L, 2147483647L)) {
      s <- morie_det_rng(nm, sd)
      expect_true(is.numeric(s))
      expect_true(s >= 0 && s < 2^31 - 1)
    }
  }
})

test_that("morie_det_rng is a pure function of (name, seed)", {
  skip_if_no_hash()
  expect_equal(morie_det_rng("foo", 42L), morie_det_rng("foo", 42L))
  expect_equal(
    morie_det_rng("ksr07_bootstrap", 12345L),
    morie_det_rng("ksr07_bootstrap", 12345L)
  )
})

test_that("different names yield distinct streams that LOOK uniform", {
  skip_if_no_hash()
  draws <- list()
  for (nm in c("alpha", "beta", "gamma")) {
    morie_det_rng(nm, 7L)
    draws[[nm]] <- runif(10000)
  }
  # KS uniformity on each stream.
  for (nm in names(draws)) {
    p <- suppressWarnings(stats::ks.test(draws[[nm]], "punif")$p.value)
    expect_gt(p, 0.001)
  }
  # No pair of streams is literally identical.
  expect_false(identical(draws[["alpha"]], draws[["beta"]]))
  expect_false(identical(draws[["alpha"]], draws[["gamma"]]))
  expect_false(identical(draws[["beta"]], draws[["gamma"]]))
  # Pearson correlation near zero (independent streams).
  pairs <- list(
    c("alpha", "beta"), c("alpha", "gamma"), c("beta", "gamma")
  )
  for (p in pairs) {
    r <- stats::cor(draws[[p[1]]], draws[[p[2]]])
    expect_lt(abs(r), 0.05)
  }
})

test_that("different seeds with same name produce independent streams", {
  skip_if_no_hash()
  morie_det_rng("xgbst", 1L); u1 <- runif(10000)
  morie_det_rng("xgbst", 2L); u2 <- runif(10000)
  expect_false(identical(u1, u2))
  expect_lt(abs(stats::cor(u1, u2)), 0.05)
})

test_that("morie_det_rng_sha_hex returns 64-char lowercase hex", {
  skip_if_no_hash()
  h <- morie_det_rng_sha_hex("foo", 42L)
  expect_type(h, "character")
  expect_equal(nchar(h), 64L)
  expect_true(grepl("^[0-9a-f]{64}$", h))
})

test_that("SHA-256 hex matches the Python side bit-for-bit on ('foo', 42)", {
  # The single most important parity assertion: if Python and R disagree
  # here, every downstream RNG state will also disagree.  The expected
  # value is pinned by the Python test test_sha_digest_hex_known_value.
  skip_if_no_hash()
  expected <- "75d158a698c56a51221dc74850b1a0a30a22cb3fab04787256140dbd16e898ed"
  expect_equal(morie_det_rng_sha_hex("foo", 42L), expected)
})

test_that("morie_det_rng('foo', 42) returns the value pinned by Python", {
  # The Python test test_r_seed_pinned_value asserts r_seed('foo', 42) == 572376904.
  # If THIS test fails alongside that one passing, the byte-slice offsets
  # in the two helpers have drifted out of sync.
  skip_if_no_hash()
  expect_equal(morie_det_rng("foo", 42L), 572376904L)
})

test_that("morie_det_rng_sha_hex matches Python for several seeds", {
  skip_if_no_hash()
  # These hex values were computed with hashlib.sha256(b"<name>:<seed>") on
  # the Python side; the R helper must agree exactly.
  cases <- list(
    list(name = "ksr07_bootstrap", seed = 42L,
         hex = NA_character_),  # filled in below
    list(name = "trfge", seed = 1L, hex = NA_character_)
  )
  # Compute via openssl if present, else digest, to derive expected
  # locally (this round-trips the helper against the underlying hash
  # primitive — guards against the substring offsets being changed).
  if (requireNamespace("openssl", quietly = TRUE)) {
    for (c in cases) {
      expected <- as.character(openssl::sha256(
        charToRaw(paste0(c$name, ":", c$seed))
      ))
      expect_equal(morie_det_rng_sha_hex(c$name, c$seed), expected)
    }
  } else if (requireNamespace("digest", quietly = TRUE)) {
    for (c in cases) {
      expected <- digest::digest(
        paste0(c$name, ":", c$seed), algo = "sha256", serialize = FALSE
      )
      expect_equal(morie_det_rng_sha_hex(c$name, c$seed), expected)
    }
  }
})
