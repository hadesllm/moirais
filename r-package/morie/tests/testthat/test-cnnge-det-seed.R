# Deterministic-seed plumbing tests for cnnge (CNN genomic predictor).

skip_if_no_hash <- function() {
  ok <- requireNamespace("digest", quietly = TRUE) ||
        requireNamespace("openssl", quietly = TRUE)
  testthat::skip_if_not(ok, "neither 'digest' nor 'openssl' available")
}

cnnge_fixture <- function() {
  set.seed(0)
  M <- matrix(stats::rnorm(15 * 8), 15, 8)
  y <- M[, 2] + M[, 4] + 0.2 * stats::rnorm(15)
  list(x = rep(0, 15), y = y, M = M)
}

test_that("cnnge deterministic_seed is reproducible", {
  skip_if_no_hash()
  fx <- cnnge_fixture()
  r1 <- cnn_genomic(fx$x, fx$y, fx$M, n_epochs = 20,
                    deterministic_seed = 42L)
  r2 <- cnn_genomic(fx$x, fx$y, fx$M, n_epochs = 20,
                    deterministic_seed = 42L)
  r3 <- cnn_genomic(fx$x, fx$y, fx$M, n_epochs = 20,
                    deterministic_seed = 999L)
  expect_equal(r1$estimate, r2$estimate)
  expect_equal(r1$W_conv, r2$W_conv)
  expect_false(isTRUE(all.equal(r1$estimate, r3$estimate)))
})

test_that("cnnge default (deterministic_seed = NULL) path is unchanged", {
  fx <- cnnge_fixture()
  r1 <- cnn_genomic(fx$x, fx$y, fx$M, n_epochs = 20, seed = 42)
  r2 <- cnn_genomic(fx$x, fx$y, fx$M, n_epochs = 20, seed = 42)
  expect_equal(r1$estimate, r2$estimate)
  expect_equal(r1$W_conv, r2$W_conv)
})
