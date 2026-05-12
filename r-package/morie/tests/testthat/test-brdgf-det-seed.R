# Deterministic-seed plumbing tests for brdgf (BayesA short Gibbs).

skip_if_no_hash <- function() {
  ok <- requireNamespace("digest", quietly = TRUE) ||
        requireNamespace("openssl", quietly = TRUE)
  testthat::skip_if_not(ok, "neither 'digest' nor 'openssl' available")
}

brdgf_fixture <- function() {
  set.seed(0)
  X <- matrix(stats::rnorm(20 * 5), 20, 5)
  b <- c(1, -1, 0.5, 0, 0)
  y <- as.numeric(X %*% b + 0.2 * stats::rnorm(20))
  list(X = X, y = y)
}

test_that("brdgf deterministic_seed is reproducible", {
  skip_if_no_hash()
  fx <- brdgf_fixture()
  r1 <- bayes_ridge_gibbs(fx$X, fx$y, n_iter = 60, burn = 20,
                          deterministic_seed = 42L)
  r2 <- bayes_ridge_gibbs(fx$X, fx$y, n_iter = 60, burn = 20,
                          deterministic_seed = 42L)
  r3 <- bayes_ridge_gibbs(fx$X, fx$y, n_iter = 60, burn = 20,
                          deterministic_seed = 999L)
  expect_equal(r1$estimate, r2$estimate)
  expect_equal(r1$beta, r2$beta)
  expect_false(isTRUE(all.equal(r1$estimate, r3$estimate)))
})

test_that("brdgf default (deterministic_seed = NULL) path is unchanged", {
  fx <- brdgf_fixture()
  r1 <- bayes_ridge_gibbs(fx$X, fx$y, n_iter = 60, burn = 20, seed = 42)
  r2 <- bayes_ridge_gibbs(fx$X, fx$y, n_iter = 60, burn = 20, seed = 42)
  expect_equal(r1$estimate, r2$estimate)
  expect_equal(r1$beta, r2$beta)
})
