# Deterministic-seed plumbing tests for dlgen (MLP genomic predictor).

skip_if_no_hash <- function() {
  ok <- requireNamespace("digest", quietly = TRUE) ||
        requireNamespace("openssl", quietly = TRUE)
  testthat::skip_if_not(ok, "neither 'digest' nor 'openssl' available")
}

dlgen_fixture <- function() {
  set.seed(0)
  M <- matrix(stats::rnorm(15 * 6), 15, 6)
  y <- M[, 1] + 0.2 * stats::rnorm(15)
  list(x = rep(0, 15), y = y, M = M)
}

test_that("dlgen deterministic_seed is reproducible", {
  skip_if_no_hash()
  fx <- dlgen_fixture()
  r1 <- deep_learning_genomic(fx$x, fx$y, fx$M, n_epochs = 30,
                              deterministic_seed = 42L)
  r2 <- deep_learning_genomic(fx$x, fx$y, fx$M, n_epochs = 30,
                              deterministic_seed = 42L)
  r3 <- deep_learning_genomic(fx$x, fx$y, fx$M, n_epochs = 30,
                              deterministic_seed = 999L)
  expect_equal(r1$estimate, r2$estimate)
  expect_equal(r1$W1, r2$W1)
  expect_false(isTRUE(all.equal(r1$estimate, r3$estimate)))
})

test_that("dlgen default (deterministic_seed = NULL) path is unchanged", {
  fx <- dlgen_fixture()
  r1 <- deep_learning_genomic(fx$x, fx$y, fx$M, n_epochs = 30, seed = 42)
  r2 <- deep_learning_genomic(fx$x, fx$y, fx$M, n_epochs = 30, seed = 42)
  expect_equal(r1$estimate, r2$estimate)
  expect_equal(r1$W1, r2$W1)
})
