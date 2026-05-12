# Deterministic-seed plumbing tests for rndsr (random search CV).
#
# Verifies the `deterministic_seed` argument added in morie v0.4.0:
#   * same deterministic_seed -> same sampled scores on repeat calls
#   * different deterministic_seed -> different sampled params/scores
#   * default deterministic_seed = NULL path is unchanged

skip_if_no_hash <- function() {
  ok <- requireNamespace("digest", quietly = TRUE) ||
        requireNamespace("openssl", quietly = TRUE)
  testthat::skip_if_not(ok, "neither 'digest' nor 'openssl' available")
}

xy_fixture <- function() {
  set.seed(0L)
  n <- 120L; p <- 3L
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  y <- as.integer(X[, 1] + X[, 2] > 0)
  list(x = X, y = y)
}

test_that("rndsr deterministic_seed is reproducible", {
  skip_if_no_hash()
  testthat::skip_if_not_installed("caret")
  d <- xy_fixture()
  r1 <- random_search_cv(d$x, d$y, n_iter = 5L, cv = 3L, deterministic_seed = 42L)
  r2 <- random_search_cv(d$x, d$y, n_iter = 5L, cv = 3L, deterministic_seed = 42L)
  r3 <- random_search_cv(d$x, d$y, n_iter = 5L, cv = 3L, deterministic_seed = 999L)
  expect_equal(r1$sampled_scores, r2$sampled_scores)
  expect_equal(r1$best_score, r2$best_score)
  expect_false(isTRUE(all.equal(r1$sampled_scores, r3$sampled_scores)))
})

test_that("rndsr default (deterministic_seed = NULL) path is unchanged", {
  testthat::skip_if_not_installed("caret")
  d <- xy_fixture()
  r1 <- random_search_cv(d$x, d$y, n_iter = 5L, cv = 3L, seed = 42L)
  r2 <- random_search_cv(d$x, d$y, n_iter = 5L, cv = 3L, seed = 42L)
  expect_equal(r1$sampled_scores, r2$sampled_scores)
  expect_equal(r1$best_score, r2$best_score)
})
