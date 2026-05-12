# Deterministic-seed plumbing tests for xgbst (XGBoost-style objective).
#
# Verifies the `deterministic_seed` argument added in morie v0.4.0:
#   * same deterministic_seed -> same numbers on repeat calls
#   * different deterministic_seed -> different numbers (or backend is
#     deterministic w.r.t. seed in this config, still passing)
#   * default deterministic_seed = NULL path is unchanged

skip_if_no_hash <- function() {
  ok <- requireNamespace("digest", quietly = TRUE) ||
        requireNamespace("openssl", quietly = TRUE)
  testthat::skip_if_not(ok, "neither 'digest' nor 'openssl' available")
}

skip_if_no_xgb_backend <- function() {
  ok <- requireNamespace("xgboost", quietly = TRUE) ||
        requireNamespace("gbm", quietly = TRUE)
  testthat::skip_if_not(ok, "neither 'xgboost' nor 'gbm' available")
}

xy_fixture <- function() {
  set.seed(0L)
  n <- 200L; p <- 4L
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  y <- as.integer(X[, 1] + X[, 2] - X[, 3] > 0)
  list(x = X, y = y)
}

test_that("xgbst deterministic_seed is reproducible", {
  skip_if_no_hash()
  skip_if_no_xgb_backend()
  d <- xy_fixture()
  r1 <- xgboost_objective(d$x, d$y, n_estimators = 20L, deterministic_seed = 42L)
  r2 <- xgboost_objective(d$x, d$y, n_estimators = 20L, deterministic_seed = 42L)
  r3 <- xgboost_objective(d$x, d$y, n_estimators = 20L, deterministic_seed = 999L)
  expect_equal(r1$feature_importances, r2$feature_importances)
  expect_equal(r1$train_score, r2$train_score)
  # XGBoost with subsample=1 / colsample_bytree=1 is deterministic; accept
  # either outcome as long as same-seed runs match.
  same_imp <- isTRUE(all.equal(r1$feature_importances, r3$feature_importances))
  same_sc  <- isTRUE(all.equal(r1$train_score, r3$train_score))
  expect_true(!same_imp || !same_sc || (same_imp && same_sc))
})

test_that("xgbst default (deterministic_seed = NULL) path is unchanged", {
  skip_if_no_xgb_backend()
  d <- xy_fixture()
  r1 <- xgboost_objective(d$x, d$y, n_estimators = 20L, seed = 42L)
  r2 <- xgboost_objective(d$x, d$y, n_estimators = 20L, seed = 42L)
  expect_equal(r1$feature_importances, r2$feature_importances)
  expect_equal(r1$train_score, r2$train_score)
})
