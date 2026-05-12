# Deterministic-seed plumbing tests for rfens (random forest ensemble).
#
# Verifies the `deterministic_seed` argument added in morie v0.4.0:
#   * same deterministic_seed -> same numbers on repeat calls
#   * different deterministic_seed -> different numbers
#   * default deterministic_seed = NULL path is unchanged

skip_if_no_hash <- function() {
  ok <- requireNamespace("digest", quietly = TRUE) ||
        requireNamespace("openssl", quietly = TRUE)
  testthat::skip_if_not(ok, "neither 'digest' nor 'openssl' available")
}

xy_fixture <- function() {
  set.seed(0L)
  n <- 200L; p <- 4L
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  y <- as.integer(X[, 1] + X[, 2] - X[, 3] > 0)
  list(x = X, y = y)
}

test_that("rfens deterministic_seed is reproducible", {
  skip_if_no_hash()
  testthat::skip_if_not_installed("randomForest")
  d <- xy_fixture()
  r1 <- random_forest_ensemble(d$x, d$y, n_estimators = 20L, deterministic_seed = 42L)
  r2 <- random_forest_ensemble(d$x, d$y, n_estimators = 20L, deterministic_seed = 42L)
  r3 <- random_forest_ensemble(d$x, d$y, n_estimators = 20L, deterministic_seed = 999L)
  expect_equal(r1$feature_importances, r2$feature_importances)
  expect_equal(r1$oob_score, r2$oob_score)
  expect_false(isTRUE(all.equal(r1$feature_importances, r3$feature_importances)))
})

test_that("rfens default (deterministic_seed = NULL) path is unchanged", {
  testthat::skip_if_not_installed("randomForest")
  d <- xy_fixture()
  r1 <- random_forest_ensemble(d$x, d$y, n_estimators = 20L, seed = 42L)
  r2 <- random_forest_ensemble(d$x, d$y, n_estimators = 20L, seed = 42L)
  expect_equal(r1$feature_importances, r2$feature_importances)
  expect_equal(r1$oob_score, r2$oob_score)
})
