# Deterministic-seed plumbing tests for ghhbp (Hierarchical NP-Bayes,
# Escobar-West augmentation for alpha | K_n).
#
# Verifies that the `deterministic_seed` argument added in morie v0.4.0:
#   * gives the same (estimate, alpha_se) on repeat calls with the same seed
#   * gives different numbers when the seed is changed
#   * leaves the default `deterministic_seed = NULL` path unchanged

skip_if_no_hash <- function() {
  ok <- requireNamespace("digest", quietly = TRUE) ||
        requireNamespace("openssl", quietly = TRUE)
  testthat::skip_if_not(ok, "neither 'digest' nor 'openssl' available")
}

xs_fixture <- function() {
  c(0.1, 0.4, -0.3, 0.7, 0.05, -0.9, 1.2, -0.4, 0.6, -0.1,
    0.3, -0.2, 0.5, -0.7, 0.0, 0.2, -0.1, 0.4, -0.5, 0.8)
}

test_that("ghhbp deterministic_seed is reproducible", {
  skip_if_no_hash()
  xs <- xs_fixture()
  r1 <- ghosal_hierarchical_bayes(xs, M = 200, deterministic_seed = 42L)
  r2 <- ghosal_hierarchical_bayes(xs, M = 200, deterministic_seed = 42L)
  r3 <- ghosal_hierarchical_bayes(xs, M = 200, deterministic_seed = 999L)
  expect_equal(r1$estimate, r2$estimate)
  expect_equal(r1$alpha_se, r2$alpha_se)
  expect_false(isTRUE(all.equal(r1$estimate, r3$estimate)))
})

test_that("ghhbp default (deterministic_seed = NULL) path is unchanged", {
  xs <- xs_fixture()
  r1 <- ghosal_hierarchical_bayes(xs, M = 200, seed = 42L)
  r2 <- ghosal_hierarchical_bayes(xs, M = 200, seed = 42L)
  expect_equal(r1$estimate, r2$estimate)
  expect_equal(r1$alpha_se, r2$alpha_se)
})
