# Deterministic-seed plumbing tests for bysid (Armstrong Ch 5
# Bayesian ideal-point estimation, Metropolis-within-Gibbs).
#
# Verifies that the `deterministic_seed` argument added in morie v0.4.0:
#   * gives the same posterior summaries on repeat calls with the same seed
#   * gives different numbers when the seed is changed
#   * leaves the default `deterministic_seed = NULL` path unchanged

skip_if_no_hash <- function() {
  ok <- requireNamespace("digest", quietly = TRUE) ||
        requireNamespace("openssl", quietly = TRUE)
  testthat::skip_if_not(ok, "neither 'digest' nor 'openssl' available")
}

vote_fixture <- function() {
  set.seed(7L)
  X <- stats::rnorm(12)
  Y <- (matrix(X, 12, 6) + matrix(stats::rnorm(12 * 6), 12, 6)) > 0
  matrix(as.numeric(Y), nrow = 12L, ncol = 6L)
}

test_that("bysid deterministic_seed is reproducible", {
  skip_if_no_hash()
  Y <- vote_fixture()
  r1 <- bysid(Y, n_iter = 120L, burn = 20L, deterministic_seed = 42L)
  r2 <- bysid(Y, n_iter = 120L, burn = 20L, deterministic_seed = 42L)
  r3 <- bysid(Y, n_iter = 120L, burn = 20L, deterministic_seed = 999L)

  expect_equal(r1$x_mean, r2$x_mean)
  expect_equal(r1$x_sd,   r2$x_sd)
  expect_equal(r1$alpha,  r2$alpha)
  expect_equal(r1$beta,   r2$beta)

  expect_false(isTRUE(all.equal(r1$x_mean, r3$x_mean)))
})

test_that("bysid default (deterministic_seed = NULL) path is unchanged", {
  Y <- vote_fixture()
  r1 <- bysid(Y, n_iter = 120L, burn = 20L, seed = 42L)
  r2 <- bysid(Y, n_iter = 120L, burn = 20L, seed = 42L)
  expect_equal(r1$x_mean, r2$x_mean)
  expect_equal(r1$x_sd,   r2$x_sd)
})
