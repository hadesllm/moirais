# Deterministic-seed plumbing tests for tsnrd (t-SNE).
#
# Verifies the `deterministic_seed` argument added in morie v0.4.0:
#   * same deterministic_seed -> same embedding on repeat calls
#   * different deterministic_seed -> different embedding
#   * default deterministic_seed = NULL path is unchanged

skip_if_no_hash <- function() {
  ok <- requireNamespace("digest", quietly = TRUE) ||
        requireNamespace("openssl", quietly = TRUE)
  testthat::skip_if_not(ok, "neither 'digest' nor 'openssl' available")
}

x_blobs_fixture <- function() {
  set.seed(0L)
  n_per <- 20L; p <- 3L
  rbind(matrix(rnorm(n_per * p, mean = -3), nrow = n_per),
        matrix(rnorm(n_per * p, mean = +3), nrow = n_per))
}

test_that("tsnrd deterministic_seed is reproducible", {
  skip_if_no_hash()
  testthat::skip_if_not_installed("Rtsne")
  X <- x_blobs_fixture()
  r1 <- tsne_reduction(X, n_components = 2L, perplexity = 5,
                       n_iter = 300L, deterministic_seed = 42L)
  r2 <- tsne_reduction(X, n_components = 2L, perplexity = 5,
                       n_iter = 300L, deterministic_seed = 42L)
  r3 <- tsne_reduction(X, n_components = 2L, perplexity = 5,
                       n_iter = 300L, deterministic_seed = 999L)
  expect_equal(r1$embedding, r2$embedding)
  expect_false(isTRUE(all.equal(r1$embedding, r3$embedding)))
})

test_that("tsnrd default (deterministic_seed = NULL) path is unchanged", {
  testthat::skip_if_not_installed("Rtsne")
  X <- x_blobs_fixture()
  r1 <- tsne_reduction(X, n_components = 2L, perplexity = 5,
                       n_iter = 300L, seed = 42L)
  r2 <- tsne_reduction(X, n_components = 2L, perplexity = 5,
                       n_iter = 300L, seed = 42L)
  expect_equal(r1$embedding, r2$embedding)
})
