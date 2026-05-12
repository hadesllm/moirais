# Deterministic-seed plumbing tests for bglup (BayesC-pi spike-and-slab).

skip_if_no_hash <- function() {
  ok <- requireNamespace("digest", quietly = TRUE) ||
        requireNamespace("openssl", quietly = TRUE)
  testthat::skip_if_not(ok, "neither 'digest' nor 'openssl' available")
}

bglup_fixture <- function() {
  set.seed(0)
  X <- matrix(stats::rnorm(30 * 6), 30, 6)
  b <- c(1, 0, 0, -1, 0, 0)
  y <- as.numeric(X %*% b + 0.1 * stats::rnorm(30))
  list(X = X, y = y)
}

test_that("bglup deterministic_seed is reproducible", {
  skip_if_no_hash()
  fx <- bglup_fixture()
  r1 <- bayes_cpi_genomic(fx$X, fx$y, n_iter = 80, burn = 30,
                          deterministic_seed = 42L)
  r2 <- bayes_cpi_genomic(fx$X, fx$y, n_iter = 80, burn = 30,
                          deterministic_seed = 42L)
  r3 <- bayes_cpi_genomic(fx$X, fx$y, n_iter = 80, burn = 30,
                          deterministic_seed = 999L)
  expect_equal(r1$estimate, r2$estimate)
  expect_equal(r1$beta, r2$beta)
  expect_false(isTRUE(all.equal(r1$estimate, r3$estimate)))
})

test_that("bglup default (deterministic_seed = NULL) path is unchanged", {
  fx <- bglup_fixture()
  r1 <- bayes_cpi_genomic(fx$X, fx$y, n_iter = 80, burn = 30, seed = 42)
  r2 <- bayes_cpi_genomic(fx$X, fx$y, n_iter = 80, burn = 30, seed = 42)
  expect_equal(r1$estimate, r2$estimate)
  expect_equal(r1$beta, r2$beta)
})
