# Deterministic-seed plumbing tests for rnnge (vanilla RNN genomic predictor).

skip_if_no_hash <- function() {
  ok <- requireNamespace("digest", quietly = TRUE) ||
        requireNamespace("openssl", quietly = TRUE)
  testthat::skip_if_not(ok, "neither 'digest' nor 'openssl' available")
}

rnnge_fixture <- function() {
  set.seed(0)
  M <- matrix(stats::rnorm(15 * 6), 15, 6)
  y <- rowSums(M) + 0.2 * stats::rnorm(15)
  list(x = rep(0, 15), y = y, M = M)
}

test_that("rnnge deterministic_seed is reproducible", {
  skip_if_no_hash()
  fx <- rnnge_fixture()
  r1 <- rnn_genomic(fx$x, fx$y, fx$M, n_epochs = 20,
                    deterministic_seed = 42L)
  r2 <- rnn_genomic(fx$x, fx$y, fx$M, n_epochs = 20,
                    deterministic_seed = 42L)
  r3 <- rnn_genomic(fx$x, fx$y, fx$M, n_epochs = 20,
                    deterministic_seed = 999L)
  expect_equal(r1$estimate, r2$estimate)
  expect_equal(r1$W_h, r2$W_h)
  expect_false(isTRUE(all.equal(r1$estimate, r3$estimate)))
})

test_that("rnnge default (deterministic_seed = NULL) path is unchanged", {
  fx <- rnnge_fixture()
  r1 <- rnn_genomic(fx$x, fx$y, fx$M, n_epochs = 20, seed = 42)
  r2 <- rnn_genomic(fx$x, fx$y, fx$M, n_epochs = 20, seed = 42)
  expect_equal(r1$estimate, r2$estimate)
  expect_equal(r1$W_h, r2$W_h)
})
