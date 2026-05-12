# Deterministic-seed plumbing tests for drpfw (inverted dropout).
#
# Verifies that the `deterministic_seed` argument added in morie v0.4.0:
#   * gives the same mask on repeat calls with the same seed
#   * gives a different mask when the seed is changed
#   * leaves the default `deterministic_seed = NULL` path unchanged

skip_if_no_hash <- function() {
  ok <- requireNamespace("digest", quietly = TRUE) ||
        requireNamespace("openssl", quietly = TRUE)
  testthat::skip_if_not(ok, "neither 'digest' nor 'openssl' available")
}

test_that("drpfw deterministic_seed is reproducible", {
  skip_if_no_hash()
  x <- rep(1, 2048L)
  r1 <- drpfw_dropout_forward(x, p = 0.5, deterministic_seed = 42L)
  r2 <- drpfw_dropout_forward(x, p = 0.5, deterministic_seed = 42L)
  r3 <- drpfw_dropout_forward(x, p = 0.5, deterministic_seed = 999L)
  expect_equal(r1$mask, r2$mask)
  expect_false(isTRUE(all.equal(r1$mask, r3$mask)))
})

test_that("drpfw default (deterministic_seed = NULL) path is unchanged", {
  x <- rep(1, 2048L)
  r1 <- drpfw_dropout_forward(x, p = 0.5, seed = 0L)
  r2 <- drpfw_dropout_forward(x, p = 0.5, seed = 0L)
  expect_equal(r1$mask, r2$mask)
})
