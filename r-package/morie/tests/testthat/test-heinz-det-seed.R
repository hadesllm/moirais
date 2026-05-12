# Deterministic-seed plumbing tests for heinz (He/Kaiming init).
#
# Verifies that the `deterministic_seed` argument added in morie v0.4.0:
#   * gives the same W on repeat calls with the same seed
#   * gives different W when the seed is changed
#   * leaves the default `deterministic_seed = NULL` path unchanged

skip_if_no_hash <- function() {
  ok <- requireNamespace("digest", quietly = TRUE) ||
        requireNamespace("openssl", quietly = TRUE)
  testthat::skip_if_not(ok, "neither 'digest' nor 'openssl' available")
}

test_that("heinz deterministic_seed is reproducible", {
  skip_if_no_hash()
  r1 <- heinz_he_initialization(64L, 32L, deterministic_seed = 42L)
  r2 <- heinz_he_initialization(64L, 32L, deterministic_seed = 42L)
  r3 <- heinz_he_initialization(64L, 32L, deterministic_seed = 999L)
  expect_equal(r1$W, r2$W)
  expect_false(isTRUE(all.equal(r1$W, r3$W)))
})

test_that("heinz default (deterministic_seed = NULL) path is unchanged", {
  r1 <- heinz_he_initialization(64L, 32L, seed = 42L)
  r2 <- heinz_he_initialization(64L, 32L, seed = 42L)
  expect_equal(r1$W, r2$W)
})
