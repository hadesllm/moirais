# Deterministic-seed plumbing tests for blasf (Bayesian LASSO short Gibbs).
#
# Verifies that the `deterministic_seed` argument added in morie v0.4.0:
#   * gives the same numeric outputs on repeat calls with the same seed
#   * gives different numbers when the seed is changed
#   * leaves the default `deterministic_seed = NULL` path unchanged

skip_if_no_hash <- function() {
  ok <- requireNamespace("digest", quietly = TRUE) ||
        requireNamespace("openssl", quietly = TRUE)
  testthat::skip_if_not(ok, "neither 'digest' nor 'openssl' available")
}

blasf_fixture <- function() {
  set.seed(0)
  X <- matrix(stats::rnorm(20 * 5), 20, 5)
  b <- c(1, -1, 0, 0, 0)
  y <- as.numeric(X %*% b + 0.2 * stats::rnorm(20))
  list(X = X, y = y)
}

test_that("blasf deterministic_seed is reproducible", {
  skip_if_no_hash()
  fx <- blasf_fixture()
  r1 <- bayesian_lasso_full(fx$X, fx$y, n_iter = 60, burn = 20,
                            deterministic_seed = 42L)
  r2 <- bayesian_lasso_full(fx$X, fx$y, n_iter = 60, burn = 20,
                            deterministic_seed = 42L)
  r3 <- bayesian_lasso_full(fx$X, fx$y, n_iter = 60, burn = 20,
                            deterministic_seed = 999L)
  expect_equal(r1$estimate, r2$estimate)
  expect_equal(r1$beta, r2$beta)
  expect_false(isTRUE(all.equal(r1$estimate, r3$estimate)))
})

test_that("blasf default (deterministic_seed = NULL) path is unchanged", {
  fx <- blasf_fixture()
  r1 <- bayesian_lasso_full(fx$X, fx$y, n_iter = 60, burn = 20, seed = 42)
  r2 <- bayesian_lasso_full(fx$X, fx$y, n_iter = 60, burn = 20, seed = 42)
  expect_equal(r1$estimate, r2$estimate)
  expect_equal(r1$beta, r2$beta)
})
