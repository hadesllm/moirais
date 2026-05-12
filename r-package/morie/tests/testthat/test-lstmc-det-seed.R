# Deterministic-seed plumbing tests for lstmc (LSTM cell forward).
#
# Verifies that the `deterministic_seed` argument added in morie v0.4.0:
#   * gives the same h on repeat calls with the same seed
#   * gives different h when the seed is changed
#   * leaves the default `deterministic_seed = NULL` path unchanged

skip_if_no_hash <- function() {
  ok <- requireNamespace("digest", quietly = TRUE) ||
        requireNamespace("openssl", quietly = TRUE)
  testthat::skip_if_not(ok, "neither 'digest' nor 'openssl' available")
}

test_that("lstmc deterministic_seed is reproducible", {
  skip_if_no_hash()
  x <- c(0.1, -0.2, 0.3, -0.4)
  r1 <- lstmc_lstm_cell(x, hidden_size = 8L, deterministic_seed = 42L)
  r2 <- lstmc_lstm_cell(x, hidden_size = 8L, deterministic_seed = 42L)
  r3 <- lstmc_lstm_cell(x, hidden_size = 8L, deterministic_seed = 999L)
  expect_equal(r1$h, r2$h)
  expect_false(isTRUE(all.equal(r1$h, r3$h)))
})

test_that("lstmc default (deterministic_seed = NULL) path is unchanged", {
  x <- c(0.1, -0.2, 0.3, -0.4)
  r1 <- lstmc_lstm_cell(x, hidden_size = 8L, seed = 0L)
  r2 <- lstmc_lstm_cell(x, hidden_size = 8L, seed = 0L)
  expect_equal(r1$h, r2$h)
})
